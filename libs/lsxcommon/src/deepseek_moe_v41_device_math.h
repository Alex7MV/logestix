// deepseek_moe_v41_device_math.h — host reference for the DeepSeek-V4.1-Flash
// bit-exact kernels (RMSNorm eps 1e-20, YaRN RoPE, Hyper-Connections with
// Sinkhorn). Plain C++ (no CUDA), shared by the kernel parity tests.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lsxcommon {
namespace deepseek_v41 {

// x[gamma]: out = gamma * x * rsqrt(mean(x^2) + eps), computed in F32.
inline void RmsNormRef(const float *x, const float *gamma, float *out, int n,
                       float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv =
        1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = gamma[i] * x[i] * inv;
}

// cos/sin tables [seqlen * (rope_dim/2)] for the reference YaRN schedule.
inline void YarnFreqsRef(int rope_dim, int seqlen, int original_seq_len,
                         float base, float factor, int beta_fast, int beta_slow,
                         std::vector<float> *cos_t, std::vector<float> *sin_t)
{
    const int half = rope_dim / 2;
    std::vector<float> freqs(static_cast<size_t>(half));
    for (int j = 0; j < half; ++j)
        freqs[j] = 1.0f / std::pow(base, static_cast<float>(2 * j) / rope_dim);
    if (original_seq_len > 0) {
        const double ln_base = std::log(static_cast<double>(base));
        auto corrected_dim = [&](double rotations) {
            return rope_dim *
                   std::log(static_cast<double>(original_seq_len) /
                            (rotations * 2.0 * M_PI)) /
                   (2.0 * ln_base);
        };
        const int low = std::max(
            static_cast<int>(std::floor(corrected_dim(beta_fast))), 0);
        const int high = std::min(
            static_cast<int>(std::ceil(corrected_dim(beta_slow))), rope_dim - 1);
        for (int i = 0; i < half; ++i) {
            const float ramp =
                std::clamp((i - low) / std::max(high - low, 1), 0, 1);
            const float smooth = 1.0f - ramp;
            freqs[i] = freqs[i] / factor * (1.0f - smooth) + freqs[i] * smooth;
        }
    }
    cos_t->assign(static_cast<size_t>(seqlen) * half, 0.0f);
    sin_t->assign(static_cast<size_t>(seqlen) * half, 0.0f);
    for (int p = 0; p < seqlen; ++p)
        for (int j = 0; j < half; ++j) {
            const float a = static_cast<float>(p) * freqs[j];
            (*cos_t)[static_cast<size_t>(p) * half + j] = std::cos(a);
            (*sin_t)[static_cast<size_t>(p) * half + j] = std::sin(a);
        }
}

// In-place rotation of the LAST rope_dim elements of x (adjacent pairs).
inline void RopeApplyRef(float *x, int rope_dim, const float *cos_row,
                         const float *sin_row, bool inverse)
{
    const int half = rope_dim / 2;
    for (int j = 0; j < half; ++j) {
        const float c = cos_row[j];
        const float s = inverse ? -sin_row[j] : sin_row[j];
        const float x0 = x[2 * j];
        const float x1 = x[2 * j + 1];
        x[2 * j] = x0 * c - x1 * s;
        x[2 * j + 1] = x0 * s + x1 * c;
    }
}

// HC coefficient split + Sinkhorn (kernel.py:407). mixes has (2+hc)*hc rows.
inline void HcSplitSinkhornRef(const float *mixes, const float *scale,
                               const float *base, int hc, int iters, float eps,
                               float *pre, float *post, float *comb)
{
    for (int j = 0; j < hc; ++j) {
        pre[j] = 1.0f / (1.0f + std::exp(-(mixes[j] * scale[0] + base[j]))) +
                 eps;
        post[j] = 2.0f /
                  (1.0f + std::exp(-(mixes[j + hc] * scale[1] + base[j + hc])));
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k)
            comb[j * hc + k] =
                mixes[j * hc + k + hc * 2] * scale[2] +
                base[j * hc + k + hc * 2];

    auto softmax_row = [&](int j) {
        float mx = comb[j * hc];
        for (int k = 1; k < hc; ++k)
            mx = std::max(mx, comb[j * hc + k]);
        float sum = 0.0f;
        for (int k = 0; k < hc; ++k) {
            comb[j * hc + k] = std::exp(comb[j * hc + k] - mx);
            sum += comb[j * hc + k];
        }
        for (int k = 0; k < hc; ++k)
            comb[j * hc + k] = comb[j * hc + k] / sum + eps;
    };
    auto norm_cols = [&]() {
        for (int k = 0; k < hc; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < hc; ++j) sum += comb[j * hc + k];
            for (int j = 0; j < hc; ++j)
                comb[j * hc + k] = comb[j * hc + k] / (sum + eps);
        }
    };
    for (int j = 0; j < hc; ++j) softmax_row(j);
    norm_cols();
    for (int it = 0; it < iters - 1; ++it) {
        for (int j = 0; j < hc; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < hc; ++k) sum += comb[j * hc + k];
            for (int k = 0; k < hc; ++k)
                comb[j * hc + k] = comb[j * hc + k] / (sum + eps);
        }
        norm_cols();
    }
}

// Collapse hc copies: out[d] = sum_c pre[c] * x[c*dim + d].
inline void HcPreRef(const float *x, const float *pre_mix, float *out, int hc,
                     int dim)
{
    for (int d = 0; d < dim; ++d) {
        float acc = 0.0f;
        for (int c = 0; c < hc; ++c) acc += pre_mix[c] * x[c * dim + d];
        out[d] = acc;
    }
}

// Expand back: out[c][d] = post[c]*x[d] + sum_cp comb[cp*hc+c]*residual[cp][d].
inline void HcPostRef(const float *x, const float *residual, const float *post,
                      const float *comb, float *out, int hc, int dim)
{
    for (int c = 0; c < hc; ++c)
        for (int d = 0; d < dim; ++d) {
            float acc = post[c] * x[d];
            for (int cp = 0; cp < hc; ++cp)
                acc += comb[cp * hc + c] * residual[cp * dim + d];
            out[c * dim + d] = acc;
        }
}

// Softmax-pool `ratio` rows of kv/score per element into out[d] (compressor,
// model.py:458). Softmax is taken over the ratio axis independently for each
// head_dim element.
inline void CompressorPoolGroupRef(const float *kv, const float *score,
                                   int ratio, int d, float *out)
{
    for (int j = 0; j < d; ++j) {
        float m = score[j];
        for (int r = 1; r < ratio; ++r)
            m = std::max(m, score[r * d + j]);
        float s = 0.0f, acc = 0.0f;
        for (int r = 0; r < ratio; ++r) {
            const float e = std::exp(score[r * d + j] - m);
            s += e;
            acc += e * kv[r * d + j];
        }
        out[j] = acc / s;
    }
}

// Round to the E2M1 grid {0,0.5,1,1.5,2,3,4,6} (nearest, ties to even index).
inline float E2m1RneRef(float v)
{
    const float tab[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = std::fabs(v);
    int best = 0;
    float bd = 1e30f;
    for (int i = 0; i < 8; ++i) {
        const float d = std::fabs(a - tab[i]);
        if (d < bd) { bd = d; best = i; }
    }
    for (int i = 0; i < 8; ++i) {
        if (i == best) continue;
        if (std::fabs(std::fabs(a - tab[i]) - bd) < 1e-6f &&
            (i % 2 == 0) && (best % 2 == 1))
            best = i;
    }
    return std::copysign(tab[best], v);
}

// 2^ceil(log2(t)) via IEEE bit inspection (matches kernel.py:fast_round_scale).
inline float FastRoundScaleRef(float t)
{
    if (t <= 0.0f) return std::ldexp(1.0f, -127);
    uint32_t bits;
    std::memcpy(&bits, &t, sizeof(bits));
    const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127 +
                    (((bits & 0x7FFFFF) != 0) ? 1 : 0);
    return std::ldexp(1.0f, exp);
}

// In-place FP4 activation quantize-dequant over `n` elements in `block` blocks.
// e8m0_scale=true -> pow2 scale, else E4M3 scale (value kept as float here).
inline void Fp4ActQuantRef(float *x, int n, int block, bool e8m0_scale)
{
    constexpr float kFp4Max = 6.0f;
    for (int b = 0; b < n; b += block) {
        float amax = 0.0f;
        for (int j = 0; j < block; ++j)
            amax = std::max(amax, std::fabs(x[b + j]));
        float s;
        if (e8m0_scale) {
            amax = std::max(amax, kFp4Max * std::ldexp(1.0f, -126));
            s = FastRoundScaleRef(amax / kFp4Max);
        } else {
            amax = std::max(amax, kFp4Max * std::ldexp(1.0f, -9));
            s = amax / kFp4Max;
        }
        for (int j = 0; j < block; ++j) {
            const float q = std::clamp(x[b + j] / s, -kFp4Max, kFp4Max);
            x[b + j] = E2m1RneRef(q) * s;
        }
    }
}

// ── Indexer top-k + candidate blocks (model.py:488,583) ────────────────────
//
// Keeps the `topk` best compressed positions (ties: lower position first),
// re-sorted into position order, shifted by `offset`; unreachable slots are -1.
// `num_valid` = number of reachable compressed positions (compress_len).
inline void IndexTopkRef(const float *score, int n_pos, int num_valid, int topk,
                         int offset, int *out)
{
    std::vector<int> idx(static_cast<size_t>(n_pos));
    for (int i = 0; i < n_pos; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] > score[b];
        return a < b;
    });
    const int k = std::min(topk, num_valid);
    std::vector<int> sel(idx.begin(), idx.begin() + k);
    std::sort(sel.begin(), sel.end());
    for (int i = 0; i < topk; ++i)
        out[i] = (i < k) ? sel[i] + offset : -1;
}

// Level-one candidate mask: block score = max over the block; the block holding
// the newest reachable position is pinned to +inf; the `topk_blocks` best blocks
// with a finite score are kept. mask has n_pos entries (1 = keep).
inline void SelectCandidateBlocksRef(const float *score, int n_pos,
                                     int num_valid, int topk_blocks,
                                     int block_size, uint8_t *mask)
{
    const int nblocks = (n_pos + block_size - 1) / block_size;
    std::vector<float> bs(static_cast<size_t>(nblocks), -INFINITY);
    for (int b = 0; b < nblocks; ++b)
        for (int j = 0; j < block_size; ++j) {
            const int p = b * block_size + j;
            if (p < n_pos) bs[b] = std::max(bs[b], score[p]);
        }
    const int last = (num_valid - 1) / block_size;
    if (last >= 0 && last < nblocks) bs[last] = INFINITY;

    std::vector<int> order(static_cast<size_t>(nblocks));
    for (int i = 0; i < nblocks; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return bs[a] > bs[b]; });

    std::fill(mask, mask + n_pos, 0);
    const int keep = std::min(topk_blocks, nblocks);
    for (int i = 0; i < keep; ++i) {
        const int b = order[i];
        if (bs[b] == -INFINITY) continue;
        for (int j = 0; j < block_size; ++j) {
            const int p = b * block_size + j;
            if (p < n_pos) mask[p] = 1;
        }
    }
}

// ── Router / Gate (model.py:809) ───────────────────────────────────────────
//
// scores are the raw gate logits (gate_weight @ x). sqrtsoftplus selects and
// scales; the noaux_tc bias only steers selection (never scales); the top-k
// weights are renormalized and multiplied by route_scale.
inline float SoftplusSqrtRef(float v)
{
    return std::sqrt(std::log1p(std::exp(v)));
}

inline void GateRouteRef(const float *scores, const float *bias, int n_routed,
                         int topk, float gate_temp, float route_scale,
                         bool norm_topk_prob, int *out_idx, float *out_w)
{
    std::vector<float> sc(static_cast<size_t>(n_routed));
    for (int i = 0; i < n_routed; ++i)
        sc[i] = SoftplusSqrtRef(scores[i] / gate_temp);

    std::vector<int> order(static_cast<size_t>(n_routed));
    for (int i = 0; i < n_routed; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const float sa = sc[a] + bias[a];
        const float sb = sc[b] + bias[b];
        if (sa != sb) return sa > sb;
        return a < b;
    });

    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        out_idx[k] = order[k];
        out_w[k] = sc[order[k]];
        sum += out_w[k];
    }
    if (norm_topk_prob && topk > 1)
        for (int k = 0; k < topk; ++k) out_w[k] /= (sum + 1e-20f);
    for (int k = 0; k < topk; ++k) out_w[k] *= route_scale;
}

// Expert SwiGLU (model.py:841): up clamped both sides, gate only above, then
// silu(gate)*up. limit <= 0 disables the clamps.
inline void ExpertSwigluRef(const float *gate, const float *up, float *out,
                            int n, float limit)
{
    for (int i = 0; i < n; ++i) {
        float u = up[i];
        float g = gate[i];
        if (limit > 0.0f) {
            u = std::clamp(u, -limit, limit);
            g = std::min(g, limit);
        }
        out[i] = (g / (1.0f + std::exp(-g))) * u;
    }
}

} // namespace deepseek_v41
} // namespace lsxcommon
