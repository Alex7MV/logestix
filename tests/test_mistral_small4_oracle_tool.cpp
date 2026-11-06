// Guard for tools/mistral_small4_oracle.py (label fast).
//
// The FSI/CODE/DET goldens for mistral-small-4-119b are declared debt because no
// source-faithful oracle can be run: the consolidated checkpoint's tensor names
// and per-expert layout do not match `Mistral4ForCausalLM`, so a plain
// `from_pretrained` loads a partially random 119B model and its forward pass
// emits NaN logits. That failure mode looks exactly like a numeric defect in the
// converted tree, and one session already lost time to it.
//
// This test pins the two things that make the tool safe to hand to the next
// session:
//   1. `load()` refuses a model whose parameters did not all come from the
//      checkpoint (`allow_missing=False` by default);
//   2. the config synthesis carries the two rope keys that each break the
//      reference forward pass when absent (`partial_rotary_factor`,
//      `llama_4_scaling_beta`), and reads geometry from the consolidated
//      field names (`dim`/`hidden_dim`/`moe.*`).
//
// It is a source scan, not an import: the tool needs torch + transformers, and
// the fast suite must not depend on the Python environment.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

std::string ReadTool()
{
    const std::filesystem::path p =
        std::filesystem::path(LSX_SOURCE_DIR) / "tools" /
        "mistral_small4_oracle.py";
    REQUIRE(std::filesystem::exists(p));
    std::ifstream f(p);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("mistral small4 oracle refuses a partially random model",
          "[tools][mistral][fast]")
{
    const std::string src = ReadTool();

    // The guard itself.
    REQUIRE(src.find("output_loading_info=True") != std::string::npos);
    REQUIRE(src.find("allow_missing") != std::string::npos);
    REQUIRE(src.find("parameters were NOT loaded from the checkpoint") !=
            std::string::npos);
    // Default must be the safe direction: a caller has to opt IN to random
    // weights, never out of the check.
    REQUIRE(src.find("def load(model_path, work_dir=None, allow_missing=False)") !=
            std::string::npos);

    // The failure mode has to be documented next to the guard, otherwise the
    // next reader deletes the check as noise.
    REQUIRE(src.find("randomly") != std::string::npos);
    REQUIRE((src.find("MISSING") != std::string::npos ||
             src.find("missing_keys") != std::string::npos));
}

TEST_CASE("mistral small4 oracle config synthesis keeps the load-bearing keys",
          "[tools][mistral][fast]")
{
    const std::string src = ReadTool();

    // Both rope keys each break the forward pass when absent, in different
    // places (rotary width vs a None multiply in the Llama-4 temperature).
    REQUIRE(src.find("\"partial_rotary_factor\"") != std::string::npos);
    REQUIRE(src.find("\"llama_4_scaling_beta\"") != std::string::npos);
    // params.json is YaRN with apply_scale = false, so the tables must not be
    // pre-multiplied.
    REQUIRE(src.find("\"rope_type\": \"yarn\"") != std::string::npos);
    REQUIRE(src.find("\"attention_factor\": 1.0") != std::string::npos);
    // Consolidated field names, not HF ones; `hidden_dim` is the dense MLP
    // width, which is the trap.
    REQUIRE(src.find("params[\"dim\"]") != std::string::npos);
    REQUIRE(src.find("params[\"hidden_dim\"]") != std::string::npos);
    REQUIRE(src.find("moe[\"expert_hidden_dim\"]") != std::string::npos);
    // The expert count / top-k come from the nested moe block.
    REQUIRE(src.find("moe[\"num_experts\"]") != std::string::npos);
    REQUIRE(src.find("moe[\"num_experts_per_tok\"]") != std::string::npos);

    // NAME_GAP is what a future implementer of the real mapping starts from.
    REQUIRE(src.find("NAME_GAP") != std::string::npos);
    REQUIRE(src.find("kv_a_proj_with_mqa") != std::string::npos);
}
