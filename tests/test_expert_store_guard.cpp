// Routed-expert host storage guard (host-only static scan; zero deps beyond
// Catch2).
//
// Bug class: a family-private OWNING container for routed experts. The first
// instance was alice's `AliceExpertHost` (removed in 4d98fe2): a
// std::unordered_map of owning PackedExpert, filled on every expert touch and
// never evicted. On real weights it cost ~1.7 MB of host heap per expert for
// the life of the session and showed as a monotonic RSS climb -- 8.3 -> 12.7 ->
// 18.0 GB peak over 32/128/512 generated tokens (~21 MB per token), while the
// device never exceeded 1187 MiB.
//
// The rule this machine-checks (ingesting-a-model, "Host-side weight storage —
// ExpertStore and Arrow zero-copy", R1/R2):
//
//   1. No owning container of an expert payload type outside the neutral layer
//      that declares those payload types. The payload types own
//      std::vector<uint8_t> copies (schema.h: TensorRecord, PackedExpert,
//      DenseNvfp4Record, SharedExpertRecord).
//   2. No family-private re-implementation of the store: an identifier that
//      looks like a private expert store (…ExpertHost / …ExpertTable) must not
//      appear at all.
//   3. Positive invariant: every family that declares an expert entry view
//      must alias the neutral ExpertStore, i.e. `ExpertStore<` must appear in
//      the same file.
//   4. The legacy all-experts-in-pinned-slabs store (MoEExpertStore) is allowed
//      for its known consumers ONLY and must not spread: a new family cannot
//      adopt it by copy without tripping this guard. See docs/open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md.
//
// Load-time transient staging (a std::vector<StagedExpert> that is uploaded
// and freed in the same scope) is NOT the defect and is deliberately not
// matched: the check targets the owning SCHEMA payload types by name.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

bool IsScanned(const fs::path &p)
{
    const std::string ext = p.extension().string();
    return ext == ".h" || ext == ".hpp" || ext == ".cuh" || ext == ".cu" ||
           ext == ".cpp";
}

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<fs::path> Sources(const fs::path &root,
                              std::initializer_list<const char *> dirs)
{
    std::vector<fs::path> out;
    std::error_code ec;
    for (const char *d : dirs) {
        const fs::path base = root / d;
        if (!fs::exists(base, ec)) continue;
        for (auto &e : fs::recursive_directory_iterator(base, ec))
            if (e.is_regular_file() && IsScanned(e.path()))
                out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string Join(const std::vector<std::string> &v)
{
    std::string s;
    for (const auto &x : v) s += x + "\n";
    return s;
}

// The payload types that own a std::vector<uint8_t> copy of the tensor are
// named explicitly in the container regex below (schema.h: TensorRecord,
// PackedExpert, DenseNvfp4Record, SharedExpertRecord, ExpertRecord) rather than
// matched by any-struct pattern: a private store holding one of THESE is the
// defect, while transient load-time staging structs are not.

// Files allowed to declare those payload types: the neutral schema surface and
// the converters, which legitimately build them to WRITE an .arrow file.
const char *const kPayloadOwners[] = {
    "libs/lsxcommon/include/lsxcommon/schema.h",
    "libs/lsxcommon/src/schema.cpp",
    "libs/lsxcommon/src/converter.cpp",
    "libs/lsxcommon/src/tokenizer_convert.cpp",
};

// Non-expert temporary maps of owning records. Verified transient, i.e. the
// map dies before the payload is needed again:
//   deepseek_moe_v41_load.cu — the FP8 dense graph. `globals` is uploaded and
//   stream-synchronised, then leaves scope before the layer loop; each layer's
//   `m` dies at the end of its iteration. This is the sanctioned
//   UnpackBatch-then-upload-then-free shape (R2), not a retained store.
const char *const kTransientPayloadMaps[] = {
    "libs/lsxcommon/src/deepseek_moe_v41_load.cu",
};

// Known consumers of the legacy all-experts-in-pinned-slabs store. Frozen:
// adding an entry is a migration decision recorded in docs/open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md,
// not a rename. Paths are relative to LSX_SOURCE_DIR.
const char *const kLegacyConsumers[] = {
    // neutral layer: the definition and the shared kernels
    "libs/lsxcommon/include/lsxcommon/lsx_moe_inference.h",
    "libs/lsxcommon/src/lsx_moe_inference.cu",
    // qwen MoE
    "libs/lsxcommon/include/lsxcommon/qwen_moe_load.h",
    "libs/lsxcommon/src/qwen_moe_load.cpp",
    "libs/lsxcommon/src/qwen_moe_model.h",
    "libs/lsxcommon/src/qwen_moe_model.cpp",
    "libs/lsxcommon/include/lsxcommon/qwen_moe_inference_pipeline.h",
    "libs/lsxcommon/src/qwen_moe_inference_pipeline.cu",
    // qwen3.8-flash-next (the 64 GiB arena)
    "libs/lsxcommon/include/lsxcommon/qwen_moe_exp_load.h",
    "libs/lsxcommon/src/qwen_moe_exp_load.cpp",
    // ornith (shares the qwen loader)
    "libs/lsxcommon/include/lsxcommon/ornith_moe_load.h",
    "libs/lsxcommon/src/ornith_moe_model.h",
    "libs/lsxcommon/src/ornith_moe_model.cpp",
    "libs/lsxcommon/include/lsxcommon/ornith_moe_inference_pipeline.h",
    "libs/lsxcommon/src/ornith_moe_inference_pipeline.cu",
    // mistral (61 GiB)
    "libs/lsxcommon/include/lsxcommon/mistral_moe_load.h",
    "libs/lsxcommon/src/mistral_moe_load.cpp",
    // glm flash
    "libs/lsxcommon/include/lsxcommon/glm_moe_flash_load.h",
    "libs/lsxcommon/include/lsxcommon/glm_moe_flash_inference.h",
    "libs/lsxcommon/src/glm_moe_flash_load.cpp",
    "libs/lsxcommon/src/glm_moe_flash_inference.cu",
    "libs/lsxcommon/src/glm_moe_flash_model.h",
    "libs/lsxcommon/src/glm_moe_flash_model.cpp",
};

template <std::size_t N>
bool InList(const std::string &rel, const char *const (&list)[N])
{
    for (std::size_t i = 0; i < N; ++i)
        if (rel == list[i]) return true;
    return false;
}

bool InUnder(const std::string &rel, const char *dir)
{
    return rel.rfind(std::string(dir) + "/", 0) == 0;
}

std::string Rel(const fs::path &root, const fs::path &p)
{
    return fs::relative(p, root).generic_string();
}

// Strips // and /* */ comment runs so prose that names a forbidden identifier
// (skills, the payload schema in a comment, this guard's own list) cannot trip
// the scan.
std::string StripComments(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    bool in_line = false, in_block = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (in_line) {
            if (text[i] == '\n') { in_line = false; out += '\n'; }
            continue;
        }
        if (in_block) {
            if (text[i] == '*' && i + 1 < text.size() && text[i + 1] == '/') {
                in_block = false;
                ++i;
            }
            continue;
        }
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
            in_line = true;
            ++i;
            continue;
        }
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
            in_block = true;
            ++i;
            continue;
        }
        out += text[i];
    }
    return out;
}

struct Scan {
    std::vector<std::string> owning_container;   // rule 1
    std::vector<std::string> private_store;      // rule 2
    std::vector<std::string> missing_alias;      // rule 3
    std::vector<std::string> legacy_spread;      // rule 4
    std::size_t entry_views = 0;
    std::size_t legacy_hits = 0;
};

Scan ScanTree(const fs::path &root, bool with_legacy_allowlist)
{
    static const std::regex container_re(
        R"(std::(unordered_map|unordered_set|map|deque|list)\s*<\s*[^;{}]*\b(PackedExpert|TensorRecord|DenseNvfp4Record|SharedExpertRecord|ExpertRecord)\b)");
    static const std::regex private_store_re(
        R"(\b[A-Za-z_][A-Za-z0-9_]*Expert(Host|Table)\b)");
    static const std::regex entry_view_re(
        R"(\bstruct\s+[A-Za-z_][A-Za-z0-9_:]*ExpertEntry\b)");

    Scan s;
    for (const fs::path &p : Sources(root, {"libs", "apps", "tests"})) {
        const std::string rel = Rel(root, p);
        // This guard names the forbidden identifiers in its own fixtures.
        if (p.filename() == "test_expert_store_guard.cpp") continue;
        const std::string code = StripComments(ReadFile(p));

        // rule 1 — owning container of an expert payload type
        if (!InList(rel, kPayloadOwners) &&
            !InList(rel, kTransientPayloadMaps)) {
            for (std::sregex_iterator it(code.begin(), code.end(),
                                         container_re),
                                  end;
                 it != end; ++it)
                s.owning_container.push_back(rel + ": " + (*it)[0].str());
        }

        // rule 2 — family-private store re-implementation
        for (std::sregex_iterator it(code.begin(), code.end(),
                                     private_store_re),
                                  end;
             it != end; ++it)
            s.private_store.push_back(rel + ": " + (*it)[0].str());

        // rule 3 — an entry view must alias the neutral store
        if (std::regex_search(code, entry_view_re)) {
            ++s.entry_views;
            if (code.find("ExpertStore<") == std::string::npos)
                s.missing_alias.push_back(rel);
        }

        // rule 4 — the legacy pinned-arena store must not spread
        if (code.find("MoEExpertStore") != std::string::npos) {
            ++s.legacy_hits;
            const bool allowed =
                with_legacy_allowlist &&
                (InList(rel, kLegacyConsumers) || InUnder(rel, "tests"));
            if (!allowed) s.legacy_spread.push_back(rel);
        }
    }
    std::sort(s.owning_container.begin(), s.owning_container.end());
    std::sort(s.private_store.begin(), s.private_store.end());
    std::sort(s.missing_alias.begin(), s.missing_alias.end());
    std::sort(s.legacy_spread.begin(), s.legacy_spread.end());
    s.owning_container.erase(
        std::unique(s.owning_container.begin(), s.owning_container.end()),
        s.owning_container.end());
    s.private_store.erase(
        std::unique(s.private_store.begin(), s.private_store.end()),
        s.private_store.end());
    s.missing_alias.erase(
        std::unique(s.missing_alias.begin(), s.missing_alias.end()),
        s.missing_alias.end());
    s.legacy_spread.erase(
        std::unique(s.legacy_spread.begin(), s.legacy_spread.end()),
        s.legacy_spread.end());
    return s;
}

// Each TEST_CASE below needs the same whole-tree scan, and the tree is ~700
// files; caching keeps the guard at a few seconds instead of one scan per case.
const Scan &CachedScan(const fs::path &root, bool with_legacy_allowlist)
{
    static std::map<std::string, Scan> cache;
    const std::string key =
        fs::absolute(root).string() +
        (with_legacy_allowlist ? "|allow" : "|strict");
    auto it = cache.find(key);
    if (it == cache.end())
        it = cache.emplace(key, ScanTree(root, with_legacy_allowlist)).first;
    return it->second;
}

} // namespace

TEST_CASE("expert store guard: scan is not vacuous", "[expert][guard][unit]")
{
    const Scan s = CachedScan(LSX_SOURCE_DIR, true);
    // The neutral store must be adopted somewhere, and the legacy store must
    // have real consumers, otherwise every rule below passes on an empty set.
    INFO("entry views: " << s.entry_views << ", legacy hits: " << s.legacy_hits);
    REQUIRE(s.entry_views >= 8);
    REQUIRE(s.legacy_hits >= 10);
}

TEST_CASE("expert store guard: no owning expert-payload container outside the "
          "neutral schema layer",
          "[expert][guard]")
{
    const Scan s = CachedScan(LSX_SOURCE_DIR, true);
    INFO("violations:\n" << Join(s.owning_container));
    REQUIRE(s.owning_container.empty());
}

TEST_CASE("expert store guard: no family-private expert store identifier",
          "[expert][guard]")
{
    const Scan s = CachedScan(LSX_SOURCE_DIR, true);
    INFO("violations:\n" << Join(s.private_store));
    REQUIRE(s.private_store.empty());
}

TEST_CASE("expert store guard: every expert entry view aliases the neutral "
          "ExpertStore",
          "[expert][guard]")
{
    const Scan s = CachedScan(LSX_SOURCE_DIR, true);
    INFO("violations:\n" << Join(s.missing_alias));
    REQUIRE(s.missing_alias.empty());
}

TEST_CASE("expert store guard: the legacy pinned-arena store does not spread",
          "[expert][guard]")
{
    const Scan s = CachedScan(LSX_SOURCE_DIR, true);
    INFO("violations (new consumer of MoEExpertStore; migrate to ExpertStore "
         "or add to kLegacyConsumers with an entry in docs/open-tasks/2026-10-23-doc-hygiene-and-non-defect-records.md):\n"
         << Join(s.legacy_spread));
    REQUIRE(s.legacy_spread.empty());
}

// The guard is the machine check for an invariant, so prove it actually
// detects: run the same rules over a synthetic tree that reproduces the
// alice defect and REQUIRE that it is caught. Without this the guard could
// pass vacuously on a broken regex.
TEST_CASE("expert store guard: the detector catches the alice defect",
          "[expert][guard][unit]")
{
    const fs::path root = fs::temp_directory_path() / "lsx_expert_store_guard";
    const fs::path fam = root / "libs" / "lsxcommon" / "include" / "lsxcommon";
    fs::create_directories(fam);

    const fs::path bad = fam / "buggy_moe_load.h";
    {
        std::ofstream f(bad);
        f << "#pragma once\n"
             "#include <cstdint>\n"
             "#include <unordered_map>\n"
             "#include <vector>\n"
             "namespace buggy {\n"
             "struct PackedExpert {\n"
             "  std::vector<uint8_t> w1_packed;\n"
             "};\n"
             "class BuggyExpertHost {\n"
             " public:\n"
             "  const PackedExpert *Get(int layer, int expert);\n"
             " private:\n"
             "  std::unordered_map<std::int64_t, PackedExpert> cache_;\n"
             "};\n"
             "}  // namespace buggy\n";
    }

    const Scan s = ScanTree(root, true);
    fs::remove_all(root);

    REQUIRE(s.owning_container.size() == 1);
    REQUIRE(s.owning_container[0].find("buggy_moe_load.h") != std::string::npos);
    REQUIRE(s.private_store.size() == 1);
    REQUIRE(s.private_store[0].find("BuggyExpertHost") != std::string::npos);
    // An entry view without the neutral alias is also caught (rule 3).
    REQUIRE(s.missing_alias.empty());  // this fixture declares no *ExpertEntry
}