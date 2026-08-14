#include "lsxnetwork/lsx_token_translator.h"
#include "lsxcommon/path_registry.h"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> ReadLines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(std::move(line));
    }
    return lines;
}

void PrintUsage() {
    SPDLOG_INFO(
        "lsx_token_packer — offline cross-model translation passport builder\n"
        "usage: lsx_token_packer --glm <vocab.txt> --deepseek <vocab.txt> "
        "--out <passport.bin>\n"
        "  [--glm-unk <id>] [--deepseek-unk <id>] [--min-overlap <k>]\n"
        "Each vocab file holds one raw BPE token per line (utf-8); the line "
        "index is the token id.");
}

}  // namespace

int main(int argc, char** argv) {
    lsxnetwork::BuildPassportOptions opts;
    std::string glm_path, ds_path, out_path =
        lsxcommon::lsx_model_metadata("passport/passport.bin");
    bool have_glm = false, have_ds = false, have_out = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                SPDLOG_ERROR("Missing value for '{}'", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--glm") {
            glm_path = next("--glm");
            have_glm = true;
        } else if (a == "--deepseek") {
            ds_path = next("--deepseek");
            have_ds = true;
        } else if (a == "--out") {
            out_path = next("--out");
            have_out = true;
        } else if (a == "--glm-unk") {
            opts.glm_unk_id = static_cast<std::uint32_t>(std::atoi(next("--glm-unk")));
        } else if (a == "--deepseek-unk") {
            opts.deepseek_unk_id = static_cast<std::uint32_t>(std::atoi(next("--deepseek-unk")));
        } else if (a == "--min-overlap") {
            opts.lmcs_min_overlap = static_cast<std::uint16_t>(std::atoi(next("--min-overlap")));
        } else {
            SPDLOG_ERROR("lsx_token_packer: unknown option '{}'", a);
            PrintUsage();
            return 2;
        }
    }

    if (!(have_glm && have_ds)) {
        PrintUsage();
        return 2;
    }
    if (!have_out)
        SPDLOG_INFO("lsx_token_packer: --out defaulted to '{}'", out_path);

    opts.glm_tokens = ReadLines(glm_path);
    opts.deepseek_tokens = ReadLines(ds_path);
    if (!out_path.empty())
        std::filesystem::create_directories(
            std::filesystem::path(out_path).parent_path());
    SPDLOG_INFO("lsx_token_packer: glm={} deepseek={} out='{}'",
                opts.glm_tokens.size(), opts.deepseek_tokens.size(), out_path);

    const int rc = lsxnetwork::WritePassport(out_path.c_str(), opts);
    if (rc != 0) {
        SPDLOG_ERROR("lsx_token_packer: failed to write '{}'", out_path);
        return 1;
    }
    SPDLOG_INFO("lsx_token_packer: passport written to '{}'", out_path);
    return 0;
}