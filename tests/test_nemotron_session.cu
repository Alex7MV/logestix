// Nemotron-3.5 device decode session (GPU; no weights, no vault).
//
// NmSession owns ~20 device buffers plus the per-layer slot maps that decide
// which layers carry Mamba-2 recurrent state and which carry a KV cache. That
// bookkeeping is derived entirely from NemotronIndex::layer_types, so a layer
// misclassified as "moe" silently loses both its state and its cache — and
// nothing else in the suite inspects it.
//
// AllocNemotronSession / ResetNemotronSession / FreeNemotronSession are covered
// here. NemotronForwardToken is not: a numeric test would need an independent
// Mamba-2 recurrence reference, which is a separate piece of work.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/nemotron_moe_inference.h"

using lsxcommon::nemotron_moe::NmSession;

namespace {

// A hybrid topology: mamba / moe / attention interleaved, as NemotronH is.
lsxcommon::nemotron_moe::NemotronIndex MakeIndex()
{
    using lsxcommon::nemotron_moe::NemotronIndex;
    NemotronIndex idx;
    idx.format = "logestix-nemotron";
    idx.model = "Nemotron-3.5-Lightning-30B-A3B";
    idx.layer_types = {"mamba", "moe", "attention", "mamba", "moe", "attention"};
    idx.num_layers = static_cast<int>(idx.layer_types.size());
    idx.hidden_size = 256;
    idx.vocab_size = 512;
    idx.mamba_num_heads = 8;
    idx.mamba_head_dim = 32;
    idx.ssm_state_size = 16;
    idx.n_groups = 4;
    idx.conv_kernel = 4;
    idx.num_attention_heads = 8;
    idx.num_key_value_heads = 2;
    idx.head_dim = 32;
    idx.num_experts = 8;
    idx.num_experts_per_tok = 2;
    idx.moe_intermediate_size = 128;
    idx.shared_expert_intermediate_size = 64;
    idx.rms_norm_eps = 1e-5;
    idx.rope_theta = 10000.0;
    return idx;
}

} // namespace

TEST_CASE("nemotron session: allocates every device buffer", "[nemotron]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    auto idx = MakeIndex();
    NmSession s;
    std::string error;
    REQUIRE(lsxcommon::nemotron_moe::AllocNemotronSession(idx, 16, s, error));
    CHECK(error.empty());

    // Derived geometry.
    CHECK(s.H == 256);
    CHECK(s.num_layers == 6);
    CHECK(s.heads == 8);
    CHECK(s.head_dim == 32);
    CHECK(s.d_inner == 8 * 32);
    CHECK(s.conv_dim == s.d_inner + 2 * 4 * 16);
    CHECK(s.max_seq == 16);

    // Every allocation the pipeline dereferences must be non-null.
    CHECK(s.h != nullptr);
    CHECK(s.normed != nullptr);
    CHECK(s.sub != nullptr);
    CHECK(s.gate != nullptr);
    CHECK(s.xbc != nullptr);
    CHECK(s.dt != nullptr);
    CHECK(s.ssm_out != nullptr);
    CHECK(s.gn_out != nullptr);
    CHECK(s.q != nullptr);
    CHECK(s.k != nullptr);
    CHECK(s.v != nullptr);
    CHECK(s.attn != nullptr);
    CHECK(s.router_logits != nullptr);
    CHECK(s.topk_w != nullptr);
    CHECK(s.topk_i != nullptr);
    CHECK(s.expert_acc != nullptr);
    CHECK(s.head_logits != nullptr);
    CHECK(s.proj16 != nullptr);
    CHECK(s.out16 != nullptr);

    lsxcommon::nemotron_moe::FreeNemotronSession(s);
}

TEST_CASE("nemotron session: layer slots separate mamba, moe and attention",
          "[nemotron]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    auto idx = MakeIndex();
    NmSession s;
    std::string error;
    REQUIRE(lsxcommon::nemotron_moe::AllocNemotronSession(idx, 8, s, error));

    REQUIRE(s.mamba_slot.size() == 6);
    REQUIRE(s.attn_slot.size() == 6);
    CHECK(s.mamba_layers == 2);
    CHECK(s.attn_layers == 2);
    // layer_types = mamba, moe, attention, mamba, moe, attention
    CHECK(s.mamba_slot[0] == 0);
    CHECK(s.mamba_slot[1] == -1);  // moe: neither
    CHECK(s.attn_slot[1] == -1);
    CHECK(s.mamba_slot[2] == -1);
    CHECK(s.attn_slot[2] == 0);
    CHECK(s.mamba_slot[3] == 1);
    CHECK(s.attn_slot[4] == -1);  // moe: neither
    CHECK(s.attn_slot[5] == 1);

    // State vectors have one entry per mamba layer and one KV entry per
    // attention layer — a moe layer must consume neither.
    REQUIRE(s.mamba_state.size() == 2);
    REQUIRE(s.conv_state.size() == 2);
    for (int i = 0; i < 2; ++i) {
        CHECK(s.mamba_state[i] != nullptr);
        CHECK(s.conv_state[i] != nullptr);
    }
    REQUIRE(s.k_cache.size() == 2);
    REQUIRE(s.v_cache.size() == 2);
    for (int i = 0; i < 2; ++i) {
        CHECK(s.k_cache[i] != nullptr);
        CHECK(s.v_cache[i] != nullptr);
    }

    lsxcommon::nemotron_moe::FreeNemotronSession(s);
}

TEST_CASE("nemotron session: Reset zeroes the recurrent and cache state",
          "[nemotron]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    auto idx = MakeIndex();
    NmSession s;
    std::string error;
    REQUIRE(lsxcommon::nemotron_moe::AllocNemotronSession(idx, 8, s, error));

    // Poison the state, then reset: a session reused for a second generation
    // run must not inherit the previous run's Mamba-2 state.
    const size_t n = static_cast<size_t>(s.heads) * s.head_dim * s.ssm_state_size;
    std::vector<float> host(n, 7.5f);
    REQUIRE(cudaMemcpy(s.mamba_state[0], host.data(), n * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    lsxcommon::nemotron_moe::ResetNemotronSession(s, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> back(n, 1.0f);
    REQUIRE(cudaMemcpy(back.data(), s.mamba_state[0], n * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (float v : back) CHECK(v == 0.0f);

    lsxcommon::nemotron_moe::FreeNemotronSession(s);
}

TEST_CASE("nemotron session: a topology with no attention layers allocates",
          "[nemotron]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    // Pure-Mamba degenerate topology: attn_layers == 0 must still allocate and
    // leave the KV vectors empty rather than indexing into them.
    auto idx = MakeIndex();
    idx.layer_types = {"mamba", "mamba"};
    idx.num_layers = 2;

    NmSession s;
    std::string error;
    REQUIRE(lsxcommon::nemotron_moe::AllocNemotronSession(idx, 4, s, error));
    CHECK(s.attn_layers == 0);
    CHECK(s.mamba_layers == 2);
    CHECK(s.k_cache.empty());
    CHECK(s.v_cache.empty());
    for (int i = 0; i < 2; ++i) CHECK(s.mamba_state[i] != nullptr);

    lsxcommon::nemotron_moe::FreeNemotronSession(s);
}

TEST_CASE("nemotron session: Free is safe and releases the buffers",
          "[nemotron]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    auto idx = MakeIndex();
    NmSession s;
    std::string error;
    REQUIRE(lsxcommon::nemotron_moe::AllocNemotronSession(idx, 4, s, error));

    lsxcommon::nemotron_moe::FreeNemotronSession(s);
    CHECK(s.h == nullptr);
    CHECK(s.k_cache.empty());
    CHECK(s.mamba_state.empty());

    // Double Free must not double-free: the model calls Free from both the
    // error path and Unload.
    lsxcommon::nemotron_moe::FreeNemotronSession(s);
    CHECK(s.h == nullptr);
    CHECK(cudaGetLastError() == cudaSuccess);
}