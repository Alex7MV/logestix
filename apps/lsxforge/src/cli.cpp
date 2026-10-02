#include "cli.h"

#include <utility>
#include <vector>

namespace lsxforge {

Resolution ResolvePipeline(const Flags &flags)
{
    // Any action flag at all means the user asked for work, so an empty
    // --input is a usage error rather than a request for the version banner.
    const bool any_action =
        flags.repack_shared || flags.repack_experts || flags.vram_recompute ||
        flags.deepseek_nvfp4 || flags.deepseek_v41 || flags.glm ||
        flags.kimi || flags.qwen || flags.qwen35moe || flags.qwen36moe ||
        flags.ornith || flags.qwen38flashnext || flags.gemma4 ||
        flags.nemotron || flags.gigachat || flags.glmflash || flags.granite ||
        flags.phi || flags.mimo || flags.alice;

    // --tokenizer supplies the source directory itself (main() assigns
    // input_path = FLAGS_tokenizer before doing any work).
    const bool has_input = !flags.input.empty() || !flags.tokenizer.empty();

    // A bare `lsxforge` prints the version banner; no work, no conversion.
    if (!any_action && !flags.run && !has_input)
        return {Pipeline::kVersion, {}};
    if (!has_input)
        return {Pipeline::kError,
                "--input is required when a conversion or side-path flag is "
                "given"};

    // --run bypasses the conversion dispatch entirely: the prompt is encoded
    // and handed to the generation loop.
    if (flags.run) return {Pipeline::kGenerate, {}};

    // Side paths, in the order main() used to test them. All of them precede
    // the tokenizer-only branch, which is why --vram-recompute wins over
    // --tokenizer rather than the pair being rejected.
    if (flags.vram_recompute) return {Pipeline::kVramRecompute, {}};
    if (flags.deepseek_nvfp4) return {Pipeline::kDeepseekNvfp4, {}};
    if (flags.deepseek_v41) return {Pipeline::kDeepseekV41, {}};
    if (flags.repack_shared) return {Pipeline::kRepackShared, {}};
    if (flags.repack_experts) return {Pipeline::kRepackRouted, {}};

    if (!flags.tokenizer.empty()) return {Pipeline::kTokenizer, {}};

    // Family flags, in the order main() tested them. The chain was
    // first-match-wins: passing two family flags silently ran the earlier one.
    // ResolvePipeline reproduces that order exactly (see
    // tests/test_lsxforge_cli.cpp "two family flags run the earlier converter")
    // — making it an error is a CLI behaviour change and belongs to the
    // operator, not to this extraction.
    const std::vector<std::pair<Pipeline, bool>> kFamilies = {
        {Pipeline::kGlmFlash, flags.glmflash},
        {Pipeline::kGranite, flags.granite},
        {Pipeline::kPhi, flags.phi},
        {Pipeline::kAlice, flags.alice},
        {Pipeline::kGemma4, flags.gemma4},
        {Pipeline::kNemotron, flags.nemotron},
        {Pipeline::kGigachat, flags.gigachat},
        {Pipeline::kMimo, flags.mimo},
        {Pipeline::kQwen38FlashNext, flags.qwen38flashnext},
        {Pipeline::kQwen36Moe, flags.qwen36moe},
        {Pipeline::kOrnith, flags.ornith},
        {Pipeline::kQwen35Moe, flags.qwen35moe},
        {Pipeline::kQwenDense, flags.qwen},
        {Pipeline::kGlm, flags.glm},
        {Pipeline::kKimi, flags.kimi},
    };
    for (const auto &[pipeline, set] : kFamilies)
        if (set) return {pipeline, {}};

    return {Pipeline::kGeneric, {}};
}

}  // namespace lsxforge