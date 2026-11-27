// TDD — GigaChat3.5 whole-stack parity (Plan 3 pre-work). Converts the
// synthetic checkpoint and compares GcPipeline::Step logits against an
// independent host-F32 reference (`test_gigachat_host_ref.h`) over a 2-token
// sequence. GPU-gated (CTest SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_moe_fixture.h"

#include "test_gigachat_host_ref.h"

#include "gigachat_moe_constants.h"
#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_inference_pipeline.h"
#include "lsxcommon/gigachat_moe_load.h"
#include "lsxcommon/schema.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace lsxcommon::gigachat;
using namespace gigachatref;

namespace {

void ForceSyntheticGeometry()
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
#endif
}

std::string LayerFile(int l, const char *suffix)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "layer_%03d%s", l, suffix);
    return std::string(buf);
}

int ArgMax(const std::vector<float> &v)
{
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

float MaxAbs(const std::vector<float> &v)
{
    float m = 0.0f;
    for (float f : v) m = std::fmax(m, std::fabs(f));
    return m;
}

RefModel LoadRefModel(const std::filesystem::path &root)
{
    const auto base = root;
    RefModel m;
    m.hidden = gigachattest::kTH;
    m.vocab = gigachattest::kTVocab;
    m.capacity = 2;
    m.eps = 1e-6f;
    m.num_experts = gigachattest::kTExperts;
    m.top_k = 1;
    m.moe_intermediate = gigachattest::kTMoeI;
    m.first_k_dense = 1;
    m.routed_scale = 2.5f;
    m.swiglu_limit = 10.0f;
    // {hidden, heads, q_lora, kv_lora, qk_nope, rope, v_head}
    m.mla_d = {gigachattest::kTH, gigachattest::kTHeads, 128, 128, 128, 64, 128};
    // {hidden, k_heads, v_heads, hk, hv, conv_k}
    m.gdn_d = {gigachattest::kTH, 1, 2, 128, 128, 4};

    auto gl = lsxcommon::LoadTensorBatchViews(base / "pinned" / "global.arrow");
    REQUIRE(gl.ok());
    m.embed = Bf16ToF32(FindTv(*gl, "model.embed_tokens.weight")->data,
                        static_cast<int64_t>(m.vocab) * m.hidden);
    m.lm_head = Bf16ToF32(FindTv(*gl, "lm_head.weight")->data,
                          static_cast<int64_t>(m.vocab) * m.hidden);
    m.final_norm = ReadNorm(*gl, "model.norm");

    for (int l = 0; l < gigachattest::kTLayers; ++l)
    {
        auto raw = lsxcommon::LoadTensorBatchViews(
            base / "pinned" / LayerFile(l, ".arrow"));
        auto nv = lsxcommon::LoadDenseNvfp4BatchViews(
            base / "pinned" / LayerFile(l, "_nvfp4.arrow"));
        REQUIRE(raw.ok());
        REQUIRE(nv.ok());
        const std::string lp = "layers." + std::to_string(l) + ".";
        RefLayer L;
        L.full_attention = (l == 1);
        L.moe = (l >= 1);
        L.input_norm = ReadNorm(*raw, lp + "input_layernorm");
        L.post_attn_norm = ReadNorm(*raw, lp + "post_attention_layernorm");
        L.post_self_attn_norm = ReadNorm(*raw, lp + "post_self_attn_layernorm");
        L.post_ffw_norm = ReadNorm(*raw, lp + "post_feedforward_layernorm");
        if (L.full_attention)
        {
            L.mla.q_a_norm = ReadNorm(*raw, lp + "self_attn.q_a_layernorm");
            L.mla.kv_a_norm = ReadNorm(*raw, lp + "self_attn.kv_a_layernorm");
            L.mla.q_a = ReadNv(*nv, lp + "self_attn.q_a_proj.weight");
            L.mla.q_b = ReadNv(*nv, lp + "self_attn.q_b_proj.weight");
            L.mla.kv_a = ReadNv(*nv, lp + "self_attn.kv_a_proj_with_mqa.weight");
            L.mla.kv_b = ReadNv(*nv, lp + "self_attn.kv_b_proj.weight");
            L.mla.o_proj = ReadNv(*nv, lp + "self_attn.o_proj.weight");
            L.mla.gate = ReadNv(*nv, lp + "self_attn.gate_proj.weight");
            L.k_cache.assign(static_cast<size_t>(m.capacity) * m.mla_d.heads *
                                 (m.mla_d.qk_nope + m.mla_d.rope),
                             0.0f);
            L.v_cache.assign(static_cast<size_t>(m.capacity) * m.mla_d.heads *
                                 m.mla_d.v_head,
                             0.0f);
        }
        else
        {
            const TensorView *conv =
                FindTv(*raw, lp + "self_attn.conv1d.weight");
            const TensorView *alog = FindTv(*raw, lp + "self_attn.A_log");
            const TensorView *dtb = FindTv(*raw, lp + "self_attn.dt_bias");
            const TensorView *ba =
                FindTv(*raw, lp + "self_attn.in_proj_ba.weight");
            const TensorView *gn = FindTv(*raw, lp + "self_attn.norm.weight");
            L.gdn.in_qkvz = ReadNv(*nv, lp + "self_attn.in_proj_qkvz.weight");
            L.gdn.out_proj = ReadNv(*nv, lp + "self_attn.out_proj.weight");
            L.gdn.conv = Bf16ToF32(conv->data, conv->shape[0] * conv->shape[2]);
            L.gdn.A_log = Bf16ToF32(alog->data, alog->shape[0]);
            L.gdn.dt_bias = Bf16ToF32(dtb->data, dtb->shape[0]);
            L.gdn.in_ba = Bf16ToF32(ba->data, ba->shape[0] * ba->shape[1]);
            L.gdn.norm = Bf16ToF32(gn->data, gn->shape[0]);
            L.gdn.conv_dim = static_cast<int>(conv->shape[0]);
            L.gdn.conv_k = static_cast<int>(conv->shape[2]);
            L.gdn.ba_rows = static_cast<int>(ba->shape[0]);
            L.gdn.v_heads = static_cast<int>(alog->shape[0]);
            L.conv_state.assign(
                static_cast<size_t>(L.gdn.conv_dim) * (L.gdn.conv_k - 1), 0.0f);
            L.rec_state.assign(
                static_cast<size_t>(L.gdn.v_heads) * m.gdn_d.hk * m.gdn_d.hv,
                0.0f);
        }
        if (L.moe)
        {
            const TensorView *router = FindTv(*raw, lp + "mlp.gate.weight");
            const TensorView *rb =
                FindTv(*raw, lp + "mlp.gate.e_score_correction_bias");
            L.moe_w.router =
                Bf16ToF32(router->data, router->shape[0] * router->shape[1]);
            L.moe_w.router_bias = ReadF32OrBf16(*rb);
            L.moe_w.num_experts = static_cast<int>(router->shape[0]);
            L.moe_w.top_k = 1;
            L.moe_w.intermediate = gigachattest::kTMoeI;
            L.moe_w.shared_gate =
                ReadNv(*nv, lp + "mlp.shared_experts.gate_proj.weight");
            L.moe_w.shared_up =
                ReadNv(*nv, lp + "mlp.shared_experts.up_proj.weight");
            L.moe_w.shared_down =
                ReadNv(*nv, lp + "mlp.shared_experts.down_proj.weight");
        }
        else
        {
            L.dense.gate = ReadNv(*nv, lp + "mlp.gate_proj.weight");
            L.dense.up = ReadNv(*nv, lp + "mlp.up_proj.weight");
            L.dense.down = ReadNv(*nv, lp + "mlp.down_proj.weight");
        }
        m.layers.push_back(std::move(L));
    }
    const float mscale =
        0.1f * std::log(static_cast<float>(frozen::kYarnFactor)) + 1.0f;
    m.attn_scale = mscale * mscale /
                   std::sqrt(static_cast<float>(m.mla_d.qk_nope + m.mla_d.rope));
    return m;
}

void AttachRefExperts(RefModel &m, GcExpertStore &store)
{
    for (int l = 0; l < gigachattest::kTLayers; ++l)
    {
        RefLayer &L = m.layers[static_cast<size_t>(l)];
        if (!L.moe) continue;
        const int E = L.moe_w.num_experts;
        const int I = L.moe_w.intermediate;
        const int H = m.hidden;
        L.moe_w.experts.assign(static_cast<size_t>(E), RefExpert{});
        for (int e = 0; e < E; ++e)
        {
            const GcExpertEntry &ent =
                store.entries[static_cast<size_t>(l) * E + e];
            RefExpert ex;
            ex.gate = {I, H, DequantNvfp4(ent.host_w1, ent.host_w1_scales,
                                          ent.w1_global_scale,
                                          static_cast<int64_t>(I) * H)};
            ex.up = {I, H, DequantNvfp4(ent.host_w3, ent.host_w3_scales,
                                        ent.w3_global_scale,
                                        static_cast<int64_t>(I) * H)};
            ex.down = {H, I, DequantNvfp4(ent.host_w2, ent.host_w2_scales,
                                          ent.w2_global_scale,
                                          static_cast<int64_t>(H) * I)};
            L.moe_w.experts[static_cast<size_t>(e)] = std::move(ex);
        }
    }
}

} // namespace

TEST_CASE("gigachat whole-stack step matches the host reference",
          "[gigachat][inference]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);
    ForceSyntheticGeometry();

    const auto src = gigachattest::MakeTempDir("gc_par_src");
    const auto root = gigachattest::MakeTempDir("gc_par_dst");
    gigachattest::WriteGigaChatCheckpoint(src, /*verified=*/true);
    const auto st = ConvertGigaChatModel(src.string(), root.string());
    INFO("convert: " << st.ToString());
    REQUIRE(st.ok());

    RefModel ref = LoadRefModel(root);

    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(root.string(), idx, err));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(GcBuildYarnTables(2, ref.mla_d.rope, ref.rc, ref.rs));

    GcWeights w;
    REQUIRE(LoadGigaChatWeights(root.string(), idx, w, stream, err));
    GcExpertCache cache;
    REQUIRE(InitGigaChatExpertCache(root.string(), idx.num_layers,
                                    idx.first_k_dense_replace, idx.num_experts,
                                    idx.num_experts_per_tok, 2, stream, cache,
                                    err));
    AttachRefExperts(ref, cache.store);

    GcPipelineDims dims;
    dims.mla = ref.mla_d;
    dims.gdn = ref.gdn_d;
    dims.vocab = ref.vocab;
    dims.num_experts = ref.num_experts;
    dims.top_k = ref.top_k;
    dims.moe_intermediate = ref.moe_intermediate;
    dims.first_k_dense = ref.first_k_dense;
    dims.rms_eps = ref.eps;
    dims.routed_scale = ref.routed_scale;
    dims.swiglu_limit = ref.swiglu_limit;

    GcPipeline pipe;
    REQUIRE(pipe.Init(w, dims, 2, err));
    REQUIRE(pipe.Reset(stream) == cudaSuccess);

    float *d_logits = nullptr;
    REQUIRE(cudaMalloc(&d_logits, static_cast<size_t>(ref.vocab) * 4) ==
            cudaSuccess);

    const std::vector<int> tokens = {1, 2};
    for (int pos = 0; pos < 2; ++pos)
    {
        std::vector<float> gpu(static_cast<size_t>(ref.vocab));
        REQUIRE(pipe.Step(w, cache, tokens[static_cast<size_t>(pos)], pos,
                          d_logits, stream) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        REQUIRE(cudaMemcpy(gpu.data(), d_logits,
                           static_cast<size_t>(ref.vocab) * 4,
                           cudaMemcpyDeviceToHost) == cudaSuccess);

        std::vector<float> host = RunModelRef(
            ref, std::vector<int>(tokens.begin(), tokens.begin() + pos + 1));
        REQUIRE(host.size() == gpu.size());
        // Error budget: the GPU GEMMs read F16 activations and store F16
        // outputs (FP32 accumulate, lsx_inference_nvfp4.cu:112), while the
        // reference computes the same dequantized weights in F32. 5e-2
        // relative is the F16-rounding envelope over two layers; the narrow
        // gates stay with the per-layer attn tests (bit-exact rope, 1e-3).
        const float tol = 5e-2f * std::fmax(1.0f, MaxAbs(host));
        float worst = 0.0f;
        for (size_t i = 0; i < host.size(); ++i)
            worst = std::fmax(worst, std::fabs(host[i] - gpu[i]));
        INFO("pos=" << pos << " argmax gpu=" << ArgMax(gpu)
                    << " ref=" << ArgMax(host) << " worst=" << worst
                    << " maxabs=" << MaxAbs(host) << " tol=" << tol);
        // Guard against a degenerate all-zero logit vector, which would make
        // the argmax comparison vacuous.
        CHECK(MaxAbs(host) > 1e-2f);
        CHECK(ArgMax(gpu) == ArgMax(host));
        CHECK(worst <= tol);
    }

    cudaFree(d_logits);
    pipe.Free();
    FreeGcExpertCache(cache);
    w.Free();
    cudaStreamDestroy(stream);
}
