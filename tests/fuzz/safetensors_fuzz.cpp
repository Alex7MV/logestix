// libFuzzer entry point for the safetensors header parser.
//
// Opt-in (see the LSX_BUILD_FUZZERS block in tests/CMakeLists.txt): the
// always-on, deterministic mutational fuzz lives in tests/test_safetensors.cpp
// so every ctest run exercises the same invariants without a corpus.
//
// The target is the header parser: it is the only code that interprets
// untrusted bytes. File-offset arithmetic is covered by the deterministic
// image-mutation test (it needs a file, which is too slow per fuzz input).
//
// Build (Clang, with the library instrumented for coverage):
//   cmake -S . -B build/fuzz -G Ninja -DCMAKE_CXX_COMPILER=clang++ \
//         -DCMAKE_BUILD_TYPE=Debug -DLSX_BUILD_FUZZERS=ON \
//         -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address -fno-omit-frame-pointer"
//   cmake --build build/fuzz --target fuzz_safetensors
//   build/fuzz/tests/fuzz_safetensors -max_total_time=60 -max_len=8192
#include <lsxcommon/safetensors.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    const std::string_view header(reinterpret_cast<const char *>(data), size);
    const auto report = lsxcommon::ParseTensorInfos(header);

    // The parser's contract: every descriptor it returns is validated and its
    // byte span is exactly numel * dtype width.
    for (const auto &t : report.tensors) {
        if (lsxcommon::ValidateTensorInfo(t) != lsxcommon::TensorInfoError::kOk)
            std::abort();
        if (t.data_end - t.data_begin != t.byte_size())
            std::abort();
    }
    return 0;
}
