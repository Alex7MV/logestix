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
        flags.deepseek_nvfp4 || flags.deepseek_v41 || flags.deepseek_pro ||
        flags.glm ||
        flags.kimi || flags.kimi_k3 || flags.kimi_k31_pro || flags.qwen ||
        flags.qwen35moe || flags.qwen36moe ||
        flags.ornith || flags.qwen38flashnext || flags.gemma4 ||
        flags.gemma4dense ||
        flags.nemotron || flags.nemotron_nvfp4 || flags.gigachat ||
        flags.glmflash || flags.granite || flags.phi || flags.dory || flags.mimo ||
        flags.alice || flags.minimax || flags.minimax_flash || flags.hy ||
        flags.hy_simulmt ||
        flags.flint ||
        flags.ling || flags.beam || flags.step || flags.inkling ||
        flags.inkling_base || flags.muse;

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
    if (flags.deepseek_pro) return {Pipeline::kDeepseekPro, {}};
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
            {"--granite_switch",
             {Pipeline::kGraniteSwitch, flags.granite_switch}},
            {"--phi", {Pipeline::kPhi, flags.phi}},
            {"--dory", {Pipeline::kDory, flags.dory}},
            {"--alice", {Pipeline::kAlice, flags.alice}},
            {"--xing", {Pipeline::kXing, flags.xing}},
            {"--mistral", {Pipeline::kMistral, flags.mistral}},
            {"--mistral_large4",
             {Pipeline::kMistralLarge4, flags.mistral_large4}},
            {"--minimax", {Pipeline::kMinimax, flags.minimax}},
            {"--minimax_flash",
             {Pipeline::kMinimaxFlash, flags.minimax_flash}},
            {"--hy", {Pipeline::kHy, flags.hy}},
            {"--hy_simulmt", {Pipeline::kHySimulMT, flags.hy_simulmt}},
            {"--flint", {Pipeline::kFlint, flags.flint}},
            {"--ling", {Pipeline::kLing, flags.ling}},
            {"--beam", {Pipeline::kBeam, flags.beam}},
            {"--step", {Pipeline::kStep, flags.step}},
            {"--gemma4", {Pipeline::kGemma4, flags.gemma4}},
            {"--gemma4dense", {Pipeline::kGemma4Dense, flags.gemma4dense}},
            {"--nemotron_nvfp4",
             {Pipeline::kNemotronNvfp4, flags.nemotron_nvfp4}},
            {"--nemotron", {Pipeline::kNemotron, flags.nemotron}},
            {"--nemotron_ultra",
             {Pipeline::kNemotronUltra, flags.nemotron_ultra}},
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
            {"--kimi_k3", {Pipeline::kKimiK3, flags.kimi_k3}},
            {"--kimi_k31_pro",
             {Pipeline::kKimiK31Pro, flags.kimi_k31_pro}},
            {"--inkling", {Pipeline::kInkling, flags.inkling}},
            {"--inkling_base", {Pipeline::kInklingBase, flags.inkling_base}},
            {"--muse", {Pipeline::kMuse, flags.muse}},
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

bool ParseLayerWindow(const std::string &layers, int &first, int &last,
                      std::string &error)
{
    first = 0;
    last = -1;  // "to the end"
    if (layers.empty()) return true;
    const auto colon = layers.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= layers.size()) {
        error = "--layers expects \"first:last\", got \"" + layers + "\"";
        return false;
    }
    const auto digits = [](const std::string &s) {
        return s.find_first_not_of("0123456789") == std::string::npos;
    };
    const std::string a = layers.substr(0, colon);
    const std::string b = layers.substr(colon + 1);
    if (!digits(a) || !digits(b)) {
        error = "--layers expects two non-negative integers, got \"" + layers + "\"";
        return false;
    }
    first = std::atoi(a.c_str());
    last = std::atoi(b.c_str());
    if (last < first) {
        error = "--layers window [" + a + "," + b + "] is inverted";
        return false;
    }
    return true;
}

}  // namespace lsxforge