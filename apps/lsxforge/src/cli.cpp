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
        flags.nemotron || flags.nemotron_nvfp4 || flags.gigachat ||
        flags.glmflash || flags.granite || flags.phi || flags.mimo ||
        flags.alice;

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
    // first-match-wins: passing two family flags silently ran the earlier one,
    // so a typo in the second flag produced a model in an unexpected layout.
    // Selecting two different converters is now rejected (exit 2).
    const std::vector<std::pair<const char *, std::pair<Pipeline, bool>>>
        kFamilies = {
            {"--glmflash", {Pipeline::kGlmFlash, flags.glmflash}},
            {"--granite", {Pipeline::kGranite, flags.granite}},
            {"--phi", {Pipeline::kPhi, flags.phi}},
            {"--alice", {Pipeline::kAlice, flags.alice}},
            {"--xing", {Pipeline::kXing, flags.xing}},
            {"--gemma4", {Pipeline::kGemma4, flags.gemma4}},
            {"--nemotron_nvfp4",
             {Pipeline::kNemotronNvfp4, flags.nemotron_nvfp4}},
            {"--nemotron", {Pipeline::kNemotron, flags.nemotron}},
            {"--gigachat", {Pipeline::kGigachat, flags.gigachat}},
            {"--mimo", {Pipeline::kMimo, flags.mimo}},
            {"--qwen38flashnext",
             {Pipeline::kQwen38FlashNext, flags.qwen38flashnext}},
            {"--qwen36moe", {Pipeline::kQwen36Moe, flags.qwen36moe}},
            {"--ornith", {Pipeline::kOrnith, flags.ornith}},
            {"--qwen35moe", {Pipeline::kQwen35Moe, flags.qwen35moe}},
            {"--qwen", {Pipeline::kQwenDense, flags.qwen}},
            {"--glm", {Pipeline::kGlm, flags.glm}},
            {"--kimi", {Pipeline::kKimi, flags.kimi}},
        };

    const char *chosen = nullptr;
    Pipeline pipeline = Pipeline::kGeneric;
    for (const auto &[name, entry] : kFamilies) {
        if (!entry.second) continue;
        if (chosen)
            return {Pipeline::kError,
                    std::string(chosen) + " and " + name +
                        " both select a converter; pass exactly one"};
        chosen = name;
        pipeline = entry.first;
    }
    if (chosen) return {pipeline, {}};

    return {Pipeline::kGeneric, {}};
}

}  // namespace lsxforge