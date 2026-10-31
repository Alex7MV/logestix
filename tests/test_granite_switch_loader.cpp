#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_switch_convert.h>
#include <lsxcommon/granite_dense_switch_load.h>

#include "test_granite_switch_fixture.h"

#include <cuda_runtime.h>

#include <filesystem>

using namespace lsxcommon::granite_switch;

TEST_CASE("granite-switch loader uploads base and adapters",
          "[granite_switch][inference]")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_switch_load_test";
    remove_all(tmp);
    create_directories(tmp);

    gswtest::BuildConverted(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    REQUIRE(ConvertGraniteSwitchModel(tmp.string(), tmp.string()).ok());

    GraniteSwitchPinnedWeights pw;
    REQUIRE(LoadGraniteSwitchPinnedWeights(tmp.string(), pw, 0));
    CHECK(pw.layers.size() == 2);
    CHECK(pw.embed.d_packed != nullptr);
    CHECK(pw.head.d_packed == pw.embed.d_packed);
    CHECK(pw.head.global_scale ==
          Catch::Approx(pw.embed.global_scale / 16.0f));
    CHECK(pw.head.view_only);

    CHECK(pw.layers[0].qkv.lora.size() == 2);
    CHECK(pw.layers[0].qkv.lora[0].size() == 3);
    CHECK(pw.layers[0].qkv.lora[0][0].rank == 4);
    CHECK(pw.layers[0].qkv.lora[1][0].rank == 8);
    CHECK(pw.layers[0].qkv.lora[1][0].d_A != nullptr);
    CHECK(pw.layers[0].qkv.lora[1][0].d_B != nullptr);

    // guardian-core targets attention only.
    CHECK(pw.layers[0].down.lora[0][0].d_A == nullptr);
    CHECK(pw.layers[0].gate_up.lora[0][0].d_A == nullptr);
    CHECK(pw.layers[0].down.lora[1][0].d_A != nullptr);
    CHECK(pw.layers[0].gate_up.lora[1][1].d_A != nullptr);

    CHECK(pw.layers[0].qkv.base.d_packed != nullptr);
    CHECK(pw.layers[0].gate_up.base.d_packed != nullptr);
    CHECK(pw.resident_bytes > 0);

    const std::uint64_t resident = pw.resident_bytes;
    pw.Free();
    CHECK(pw.embed.d_packed == nullptr);
    CHECK(pw.head.d_packed == nullptr);
    CHECK(pw.layers.empty());
    CHECK(pw.resident_bytes == 0);
    CHECK(resident > 0);

    remove_all(tmp);
}
