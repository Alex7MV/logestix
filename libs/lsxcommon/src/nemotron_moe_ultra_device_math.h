#pragma once

// Host reference math for NVIDIA-Nemotron-3-Ultra-550B-A55B.
//
// These are the formulas the device kernels are validated against. They are
// transcribed from the reference implementation (transformers 5.15.1):
//   models/nemotron_h/modeling_nemotron_h.py
//     NemotronHTopkRouter.forward   L728-753
//     NemotronHMoE.forward          L699-712
//     NemotronHMamba2Mixer.forward  L427-571
//     NemotronHRMSNorm.forward      L584-589
//   models/zamba2/modeling_zamba2.py
//     Zamba2RMSNormGated
//     mamba2_selective_state_update L478-534  (decode: softplus, NO dt clamp)
//     mamba2_chunk_scan             L540-562  (prefill: softplus THEN clamp)
//
// Host-only and CUDA-free on purpose: it is the oracle for tests and the
// conversion-side sanity instrument, never linked into the device path.
//
// Two pieces are specific to this model and do not exist in the 30B sibling:
// the GROUPED router (n_group 8 / topk_group 1) and the LATENT MoE (the routed
// experts run at moe_latent_size, not at hidden).

#include <cmath>
#include <cstddef>

#include "nemotron_moe_ultra_constants.h"

namespace lsxcommon::nemotron_ultra {
namespace ref {

inline float Silu(float x) { return x / (1.0f + std::exp(-x)); }

inline float Softplus(float x)
{
    // numerically stable log(1 + e^x)
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline float Relu2(float x)
{
    const float r = x > 0.0f ? x : 0.0f;
    return r * r;
}

// RMSNorm without mean subtraction or bias (NemotronHRMSNorm): computed in F32,
// applied to the F32 activation the engines carry internally.
inline void RmsNorm(const float *x, const float *weight, int n, float eps,
                    float *out)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = weight[i] * x[i] * inv;
}

// Gated RMSNorm (Zamba2RMSNormGated): h = x * silu(gate); per-group rmsnorm;
// final elementwise weight. Layouts are flat: x/gate/weight are [n].
inline void GatedRmsNorm(const float *x, const float *gate, const float *weight,
                         int n, int group_size, float eps, float *out)
{
    const int groups = n / group_size;
    for (int g = 0; g < groups; ++g) {
        const int base = g * group_size;
        double ss = 0.0;
        for (int i = 0; i < group_size; ++i) {
            const float h = x[base + i] * Silu(gate[base + i]);
            ss += static_cast<double>(h) * static_cast<double>(h);
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / group_size) + eps);
        for (int i = 0; i < group_size; ++i) {
            const float h = x[base + i] * Silu(gate[base + i]);
            out[base + i] = weight[base + i] * h * inv;
        }
    }
}

// One step of the Mamba-2 selective state-space recurrence.
//   xs       [heads, head_dim]        post-conv activation
//   B, C     [groups, state_size]
//   D        [heads]
//   dt       [heads]                  raw in_proj slice
//   dt_bias  [heads], A_log [heads]
//   state    [heads, head_dim, state_size]  (in/out, carried across steps)
//   out      [heads, head_dim]
//
// clamp_dt mirrors the reference split exactly: the prefill chunk scan clamps dt
// to time_step_min, while the single-token selective update applies softplus
// only (see the reference lines above). Getting this backwards shifts every
// prefill state by up to the clamp floor.
inline void SelectiveStateUpdate(const float *state_in, const float *xs,
                                 const float *B, const float *C, const float *D,
                                 const float *dt, const float *dt_bias,
                                 const float *A_log, int heads, int head_dim,
                                 int groups, int state_size, bool clamp_dt,
                                 float *state_out, float *out)
{
    const int heads_per_group = heads / groups;
    for (int h = 0; h < heads; ++h) {
        float dth = Softplus(dt[h] + dt_bias[h]);
        if (clamp_dt && dth < frozen::kDtMin) dth = frozen::kDtMin;
        const float A = -std::exp(A_log[h]);
        const float dA = std::exp(dth * A);
        const int g = h / heads_per_group;
        const float *Bg = B + static_cast<std::size_t>(g) * state_size;
        const float *Cg = C + static_cast<std::size_t>(g) * state_size;
        for (int d = 0; d < head_dim; ++d) {
            const float xsd = xs[h * head_dim + d];
            const std::size_t st_base =
                (static_cast<std::size_t>(h) * head_dim + d) * state_size;
            float acc = 0.0f;
            for (int s = 0; s < state_size; ++s) {
                const float st = state_in[st_base + s] * dA + dth * Bg[s] * xsd;
                state_out[st_base + s] = st;
                acc += st * Cg[s];
            }
            out[h * head_dim + d] = acc + xsd * D[h];
        }
    }
}

// Grouped top-k router (NemotronHTopkRouter), the Ultra form:
//   scores           = sigmoid(logits)
//   scores_for_choice= scores + e_score_correction_bias      (bias selects only)
//   group_scores[g]  = sum of the top-2 scores_for_choice in group g
//   keep the `topk_group` groups with the highest group score, mask the rest
//   idx  = top-k of the surviving scores_for_choice
//   w    = scores[idx] / sum(scores[idx]) * scale            (norm_topk_prob)
//
// Ties resolve to the lowest expert index, which is what makes the selection
// reproducible against the device kernel; the reference's topk on a tie is
// implementation-defined, so the kernel is the authority and this is its mirror.
inline void RouterTopKGrouped(const float *logits, const float *e_score_bias, int E,
                              int k, int n_group, int topk_group, float scale,
                              float *weights, int *indices)
{
    const int per_group = E / n_group;
    float scores[frozen::kNumRouted];
    float choice[frozen::kNumRouted];
    for (int e = 0; e < E; ++e) {
        scores[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        choice[e] = scores[e] + e_score_bias[e];
    }

    // Group scores: sum of the top two within each group. per_group may be 1
    // (topk_group over singletons), in which case the score is that one value.
    float group_score[frozen::kNumRouted];
    for (int g = 0; g < n_group; ++g) {
        const int base = g * per_group;
        float best1 = -INFINITY, best2 = -INFINITY;
        for (int i = 0; i < per_group; ++i) {
            const float v = choice[base + i];
            if (v > best1) {
                best2 = best1;
                best1 = v;
            } else if (v > best2) {
                best2 = v;
            }
        }
        group_score[g] = best1 + (per_group > 1 ? best2 : 0.0f);
    }

    // Keep the best `topk_group` groups (lowest index wins a tie).
    bool group_kept[frozen::kNumRouted];
    for (int g = 0; g < n_group; ++g) group_kept[g] = false;
    for (int pick = 0; pick < topk_group; ++pick) {
        int best = -1;
        for (int g = 0; g < n_group; ++g) {
            if (group_kept[g]) continue;
            if (best < 0 || group_score[g] > group_score[best]) best = g;
        }
        if (best >= 0) group_kept[best] = true;
    }
    for (int g = 0; g < n_group; ++g)
        if (!group_kept[g])
            for (int i = 0; i < per_group; ++i)
                choice[g * per_group + i] = -INFINITY;

    // Top-k over the survivors, lowest index on a tie.
    for (int t = 0; t < k; ++t) {
        int best = -1;
        for (int e = 0; e < E; ++e) {
            bool taken = false;
            for (int p = 0; p < t; ++p)
                if (indices[p] == e) taken = true;
            if (taken) continue;
            if (best < 0 || choice[e] > choice[best]) best = e;
        }
        indices[t] = best;
    }

    float sum = 0.0f;
    for (int t = 0; t < k; ++t) {
        weights[t] = scores[indices[t]];
        sum += weights[t];
    }
    const float denom = sum + 1e-20f;
    for (int t = 0; t < k; ++t) weights[t] = weights[t] / denom * scale;
}

// Non-gated relu2 expert MLP at the LATENT width: out = down @ relu2(up @ x).
//   x [in], up [i, in], down [in, i], out [in]
inline void ExpertMlpRelu2(const float *x, const float *up, const float *down,
                           int in, int i, float *out)
{
    float act[frozen::kMoeIntermediate];
    for (int r = 0; r < i; ++r) {
        float acc = 0.0f;
        const float *row = up + static_cast<std::size_t>(r) * in;
        for (int c = 0; c < in; ++c) acc += row[c] * x[c];
        act[r] = Relu2(acc);
    }
    for (int r = 0; r < in; ++r) {
        float acc = 0.0f;
        const float *row = down + static_cast<std::size_t>(r) * i;
        for (int c = 0; c < i; ++c) acc += row[c] * act[c];
        out[r] = acc;
    }
}

// The latent MoE (NemotronHMoE.forward): the routed experts run at `L` and the
// result is projected back to `H`; the shared expert runs at `H` and is added
// after the projection, never inside the latent space.
//   x          [H]
//   fc1        [L, H]      latent down-projection
//   fc2        [H, L]      latent up-projection
//   shared_up  [SI, H]     shared expert up (wide, at H)
//   shared_down[H, SI]     shared expert down
//   expert_up  [topk][I, L]   per tapped expert, at the LATENT width
//   expert_down[topk][L, I]
//   expert_ids [topk], expert_w [topk]
//   out        [H]
inline void LatentMoe(const float *x, const float *fc1, const float *fc2,
                      const float *shared_up, const float *shared_down,
                      const float *const *expert_up,
                      const float *const *expert_down, const int *expert_ids,
                      const float *expert_w, int topk, int H, int L, int I,
                      int SI, float *out)
{
    float latent[frozen::kMoeLatent];
    for (int r = 0; r < L; ++r) {
        float acc = 0.0f;
        const float *row = fc1 + static_cast<std::size_t>(r) * H;
        for (int c = 0; c < H; ++c) acc += row[c] * x[c];
        latent[r] = acc;
    }

    float acc_latent[frozen::kMoeLatent];
    for (int r = 0; r < L; ++r) acc_latent[r] = 0.0f;
    for (int t = 0; t < topk; ++t) {
        (void)expert_ids;  // the caller already indexed the tapped weights
        float y[frozen::kMoeIntermediate];
        ExpertMlpRelu2(latent, expert_up[t], expert_down[t], L, I, y);
        for (int r = 0; r < L; ++r) acc_latent[r] += expert_w[t] * y[r];
    }

    // Latent -> hidden.
    for (int r = 0; r < H; ++r) {
        float acc = 0.0f;
        const float *row = fc2 + static_cast<std::size_t>(r) * L;
        for (int c = 0; c < L; ++c) acc += row[c] * acc_latent[c];
        out[r] = acc;
    }

    // Shared expert at the HIDDEN width, added after the projection.
    float sact[frozen::kSharedIntermediate];
    for (int r = 0; r < SI; ++r) {
        float acc = 0.0f;
        const float *row = shared_up + static_cast<std::size_t>(r) * H;
        for (int c = 0; c < H; ++c) acc += row[c] * x[c];
        sact[r] = Relu2(acc);
    }
    for (int r = 0; r < H; ++r) {
        float acc = 0.0f;
        const float *row = shared_down + static_cast<std::size_t>(r) * SI;
        for (int c = 0; c < SI; ++c) acc += row[c] * sact[c];
        out[r] += acc;
    }
}

} // namespace ref
} // namespace lsxcommon::nemotron_ultra
