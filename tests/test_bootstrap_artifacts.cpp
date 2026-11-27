// The codec contract on REAL BYTES (plan Task 9a, step 2).
//
// Every other test of this contract runs on a synthetic tree it wrote itself, so
// it proves the code agrees with itself. This one reads the two artifacts the
// bootstrap conversion produced from real checkpoints and asks the questions a
// fixture cannot answer:
//
//   * does the mask a real converter wrote describe the bytes beside it, and does
//     it survive a re-digest of the whole tree (VerifyArtifact deep)?
//   * does the artifact mirror the store path it was converted from, so moving
//     between a checkpoint and its artifacts means changing only the root?
//   * is it self-sufficient — tokenizer and chat template INSIDE the profile dir,
//     loadable with LSX_SOURCE_ROOT unmounted (invariant I5, design §1.5)?
//   * does the admission gate refuse it on silicon that cannot execute NVFP4,
//     while admitting it on the card it was converted for?
//
// The self-sufficiency question is why this file exists: a scan of the 30 trees
// that predated the contract found chat_template.jinja in 0 of them, and a grep
// of the emitter's call site found the reason — it looked for the template under
// the ARTIFACT dir, where no converter ever writes it (design §1.5 measured this,
// and the copy step was dead code until this test caught it). A fixture cannot
// catch that: the fixture is the thing that puts the file where the code looks.
//
// A SKIP is not a pass (AGENTS.md). The cold start is recorded in
// docs/OPEN_TASKS.md; this test is the reason that record has to say "not run"
// rather than nothing at all.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/hardware_target.h"
#include "lsxstorage/artifact.h"
#include "lsxstorage/roots.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace lsx::storage;
using namespace lsxcommon::hw;

namespace {

struct Bootstrap {
    const char *rel;        // <root>/<rel>
    const char *org;        // the store's own spelling
    const char *model_dir;
};

// The two slice models of the bootstrap conversion. Deliberately hardcoded: the
// point is that these EXACT trees, converted from real checkpoints, satisfy the
// contract. A list derived from the store would pass vacuously the day the store
// is empty.
constexpr Bootstrap kBootstraps[] = {
    {"microsoft/phi-4/nvfp4", "microsoft", "phi-4"},
    {"Qwen/Qwen3.5-9B/nvfp4", "Qwen", "Qwen3.5-9B"},
};

// Skips the whole case when the store or a bootstrap artifact is absent, naming
// what is missing — "the test passed" and "the test had nothing to read" must not
// look the same in the ctest log.
void RequireStoreAndArtifacts()
{
    const auto root = ArtifactRoot();
    if (!root)
        SKIP("LSX_ARTIFACT_ROOT unset or not a usable path — the bootstrap "
             "conversion has not been run");
    std::vector<std::string> missing;
    for (const auto &b : kBootstraps)
        if (!fs::is_directory(*root / b.rel))
            missing.push_back((*root / b.rel).string());
    if (!missing.empty()) {
        for (const auto &m : missing)
            WARN("missing artifact: " + m);
        SKIP("bootstrap artifacts absent — run lsxforge --<family> "
             "--codec-profile nvfp4 for each (plan Task 9a step 1)");
    }
}

ArtifactInfo ReadBootstrap(const Bootstrap &b)
{
    const auto dir = *ArtifactRoot() / b.rel;
    const auto r = ReadArtifact(dir);
    INFO("artifact: " << dir.string() << " — " << r.detail);
    REQUIRE(r.status == ReadStatus::kOk);
    return r.info;
}

} // namespace

TEST_CASE("a bootstrap artifact mirrors the store path it was converted from",
          "[bootstrap][artifact]") {
    RequireStoreAndArtifacts();
    for (const auto &b : kBootstraps) {
        INFO("artifact: " << b.rel);
        const auto info = ReadBootstrap(b);
        // Identity, in the store's spelling rather than the manifest's: an
        // operator navigates by what `ls` shows, and `<org>/<model_dir>/<profile>`
        // is the contract that makes the artifact store a mirror of the source
        // store (design §2.4.2).
        CHECK(info.org == b.org);
        CHECK(info.model_dir == b.model_dir);
        CHECK(info.profile == "nvfp4");
        CHECK_FALSE(info.model_type.empty());
        CHECK_FALSE(info.family.empty());
        // An artifact of unknown provenance cannot be gated: the reader already
        // refuses one, and this pins that it refused for the stated reason.
        CHECK_FALSE(info.source_dir.empty());
    }
}

TEST_CASE("a bootstrap artifact declares exactly the codecs a dense NVFP4 "
          "profile ships",
          "[bootstrap][artifact]") {
    RequireStoreAndArtifacts();
    // F32/bf16 are the un-quantized edges (norm weights, the tied table, the
    // router) and NVFP4 is the body. Anything else in the mask would mean the
    // converter wrote a plane the profile never declared — the "second source of
    // truth" failure the mask exists to prevent.
    constexpr CodecMask kExpectedOnly = Mask(PlaneCodec::kF32) |
                                        Mask(PlaneCodec::kBF16) |
                                        Mask(PlaneCodec::kNvFp4);
    for (const auto &b : kBootstraps) {
        INFO("artifact: " << b.rel);
        const auto info = ReadBootstrap(b);
        CHECK((info.used_codecs & Mask(PlaneCodec::kNvFp4)) != 0);
        CHECK((info.used_codecs & ~kExpectedOnly) == 0);
        CHECK(info.used_codecs != 0);
        // The per-layer map is what a loader asks "which codec is layer N?"; an
        // empty or short map makes that question unanswerable at runtime, and a
        // zero entry would name no codec at all.
        REQUIRE(info.layer_count > 0);
        CHECK(info.layer_codec.size() ==
              static_cast<size_t>(info.layer_count));
        for (size_t i = 0; i < info.layer_codec.size(); ++i) {
            INFO("layer " << i << " codec mask " << info.layer_codec[i]);
            CHECK(info.layer_codec[i] != 0);
        }
        // Every plane that carries bytes must name its codec, except role
        // "meta" (the tokenizer arrows, an index table): those establish no
        // codec, which is the rule EmitArtifact enforces on the writer side.
        for (const auto &p : info.planes) {
            INFO("plane " << p.path << " role " << p.role);
            if (p.role == "meta") {
                CHECK(p.codec_bit == 0);
                continue;
            }
            CHECK(p.codec_bit != 0);
            CHECK_FALSE(p.path.empty());
            CHECK(p.bytes > 0);
            CHECK(p.has_digest);
        }
    }
}

TEST_CASE("a bootstrap artifact is self-sufficient: tokenizer and template "
          "inside",
          "[bootstrap][artifact]") {
    RequireStoreAndArtifacts();
    for (const auto &b : kBootstraps) {
        INFO("artifact: " << b.rel);
        const auto dir = *ArtifactRoot() / b.rel;
        const auto info = ReadBootstrap(b);
        // The tokenizer the loaders actually read: two Arrow tables inside the
        // artifact. Both must be recorded AND present, because a recorded path
        // that is missing turns into a load failure at the first prompt.
        REQUIRE_FALSE(info.tokenizer_vocab.empty());
        REQUIRE_FALSE(info.tokenizer_merges.empty());
        CHECK(fs::is_regular_file(dir / info.tokenizer_vocab));
        CHECK(fs::is_regular_file(dir / info.tokenizer_merges));
        // The chat template is the file the emitter's copy step exists for. Both
        // bootstrap checkpoints ship one, so an empty record here means the copy
        // silently did not happen — which is precisely the defect this case was
        // written to catch (it was red before the fix: the emitter looked under
        // the artifact dir instead of the checkpoint).
        INFO("recorded chat template: '" << info.chat_template << "'");
        CHECK_FALSE(info.chat_template.empty());
        CHECK(fs::is_regular_file(dir / info.chat_template));
    }
}

TEST_CASE("the admission gate refuses a bootstrap NVFP4 artifact on silicon "
          "without native FP4",
          "[bootstrap][gate]") {
    RequireStoreAndArtifacts();
    for (const auto &b : kBootstraps) {
        INFO("artifact: " << b.rel);
        const auto used = ReadBootstrap(b).used_codecs;

        // Blackwell is the card these were converted for: admitted.
        CHECK(CheckCodecs(used, TargetFromComputeCapability(12, 0)).allowed);
        CHECK(CheckCodecs(used, TargetFromComputeCapability(12, 1)).allowed);

        // Ampere has no FP4 tensor-core op. This is invariant I1 in its only
        // observable form: the refusal must come from the mask the artifact
        // really carries, not from a hand-written constant in a test.
        for (const auto mm : {std::pair{8, 6}, std::pair{8, 7}}) {
            const auto d =
                CheckCodecs(used, TargetFromComputeCapability(mm.first, mm.second));
            INFO("sm_" << mm.first << mm.second << ": " << d.message);
            CHECK_FALSE(d.allowed);
            CHECK((d.missing & Mask(PlaneCodec::kNvFp4)) != 0);
            // The operator reads this message, so it must NAME the codec that is
            // missing rather than say "unsupported target".
            CHECK(d.message.find("nvfp4") != std::string::npos);
        }

        // A CPU target is the fallback when no device is present, and it must
        // fail closed too: the gate is a capability question, not a vendor one.
        const auto cpu =
            CheckCodecs(used, TargetFromCpuFeatures(true, true, true, true, false,
                                                    false));
        CHECK_FALSE(cpu.allowed);
        CHECK((cpu.missing & Mask(PlaneCodec::kNvFp4)) != 0);
    }
}

TEST_CASE("a bootstrap artifact survives a re-digest of every byte it ships",
          "[bootstrap][verify]") {
    RequireStoreAndArtifacts();
    for (const auto &b : kBootstraps) {
        INFO("artifact: " << b.rel);
        const auto dir = *ArtifactRoot() / b.rel;
        const auto report = VerifyArtifact(dir, /*deep=*/true);
        for (const auto &p : report.problems) INFO("problem: " << p);
        CHECK(report.ok);
        // The mask is a union recomputed from the files present, and it must
        // equal what artifact.fb claims — a drift here ships an artifact that
        // declares one codec and contains another.
        CHECK(report.declared_mask == report.actual_mask);
        CHECK(report.declared_mask != 0);
        CHECK(report.declared_mask == ReadArtifact(dir).info.used_codecs);
        // v1 ships no ballast (design §1.4). Qwen3.5-9B carries
        // `mtp_num_hidden_layers: 1` in its config, so this is a real assertion
        // for that checkpoint and not a formality.
        CHECK_FALSE(fs::exists(dir / "mtp"));
        CHECK_FALSE(fs::exists(dir / "deferred"));
    }
}
