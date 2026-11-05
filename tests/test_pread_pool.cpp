// Persistent pread pool behavior (host-only, no GPU, no checkpoint).
//
// The pool exists because a single-threaded pread loop over NVMe is
// latency-bound: one stream measures ~2.5 GB/s while the array does ~39.8 GB/s
// at 32 concurrent cold reads. The expert staging ring fills its pinned slots
// through this pool, so a bug here is a partially filled staging slot — i.e.
// silently wrong weights.
#include <catch2/catch_test_macros.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/pread_pool.h"

namespace {

// One file per "expert", each filled with its own byte value so a job that
// reads the wrong file, or the wrong offset, is visible rather than plausible.
struct Fixture {
    std::filesystem::path dir;
    std::vector<std::string> paths;

    Fixture(int n, size_t bytes)
    {
        dir = std::filesystem::temp_directory_path() /
              ("lsx_pread_pool_test_" + std::to_string(::getpid()));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        for (int i = 0; i < n; ++i) {
            const auto p = (dir / ("f" + std::to_string(i))).string();
            std::vector<uint8_t> buf(bytes, static_cast<uint8_t>(i + 1));
            const int fd = ::open(p.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            REQUIRE(fd >= 0);
            REQUIRE(::write(fd, buf.data(), buf.size()) ==
                    static_cast<ssize_t>(buf.size()));
            ::close(fd);
            paths.push_back(p);
        }
    }
    ~Fixture() { std::filesystem::remove_all(dir); }
};

}  // namespace

TEST_CASE("PreadPool fills every job from its own file", "[pread_pool]")
{
    Fixture fx(8, 64 * 1024);
    lsxcommon::io::PreadPool pool(4);
    std::vector<std::vector<uint8_t>> out(8, std::vector<uint8_t>(64 * 1024));
    std::vector<lsxcommon::io::PreadJob> jobs;
    std::vector<int> fds;
    for (int i = 0; i < 8; ++i) {
        const int fd = ::open(fx.paths[static_cast<size_t>(i)].c_str(), O_RDONLY);
        REQUIRE(fd >= 0);
        fds.push_back(fd);
        jobs.push_back({fd, 0, out[static_cast<size_t>(i)].data(),
                        out[static_cast<size_t>(i)].size()});
    }
    REQUIRE(pool.Run(jobs) == 8);
    for (int i = 0; i < 8; ++i)
        REQUIRE(out[static_cast<size_t>(i)][0] ==
                static_cast<uint8_t>(i + 1));
    for (int fd : fds) ::close(fd);
}

TEST_CASE("PreadPool reads at an offset, not from the file head", "[pread_pool]")
{
    Fixture fx(1, 4096);
    // Fill the destination with 0xEE first: a job that never wrote would leave
    // the sentinel and pass a "not zero" check, so the assertion is the exact
    // file byte.
    lsxcommon::io::PreadPool pool(1);
    std::vector<uint8_t> out(1024, 0xEE);
    const int fd = ::open(fx.paths[0].c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    std::vector<lsxcommon::io::PreadJob> jobs{{fd, 512, out.data(), out.size()}};
    REQUIRE(pool.Run(jobs) == 1);
    REQUIRE(out[0] == 0x01);
    REQUIRE(out[1023] == 0x01);
    ::close(fd);
}

TEST_CASE("PreadPool does not count a short read as complete", "[pread_pool]")
{
    Fixture fx(1, 4096);
    lsxcommon::io::PreadPool pool(2);
    std::vector<uint8_t> out(8192, 0xEE);
    const int fd = ::open(fx.paths[0].c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    // Ask for more than the file holds. pread returns a short read; counting it
    // as complete would DMA a half-filled staging slot to the device.
    std::vector<lsxcommon::io::PreadJob> jobs{{fd, 0, out.data(), out.size()}};
    REQUIRE(pool.Run(jobs) == 0);
    // The bytes it could read are there; what matters is the return value, not
    // the buffer, because the caller falls back on an incomplete job.
    REQUIRE(out[0] == 0x01);
    ::close(fd);
}

TEST_CASE("PreadPool with a bad descriptor completes nothing", "[pread_pool]")
{
    lsxcommon::io::PreadPool pool(2);
    std::vector<uint8_t> out(64, 0xEE);
    std::vector<lsxcommon::io::PreadJob> jobs{{-1, 0, out.data(), out.size()}};
    REQUIRE(pool.Run(jobs) == 0);
}

TEST_CASE("PreadPool defaults to four threads when asked for none",
          "[pread_pool]")
{
    // The measured default: 4 threads reached 13.9 GB/s against a 39.8 GB/s
    // array ceiling, and the sweep that finds the knee is a later task, so the
    // construction-time default has to be a real number, not 0.
    lsxcommon::io::PreadPool pool(0);
    REQUIRE(pool.Threads() == 4);
    lsxcommon::io::PreadPool neg(-3);
    REQUIRE(neg.Threads() == 4);
    lsxcommon::io::PreadPool two(2);
    REQUIRE(two.Threads() == 2);
}
