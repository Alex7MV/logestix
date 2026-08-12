#include "lsxnetwork/affinity.h"

// pthread_setaffinity_np / pthread_getaffinity_np are GNU extensions; without
// _GNU_SOURCE the declarations are hidden on glibc and the Linux build breaks.
#define _GNU_SOURCE 1

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <pthread.h>
#include <sched.h>
#endif

namespace lsxnetwork {

int CpuAffinity::PinToCores0To3() noexcept {
#if defined(_WIN32)
    const DWORD_PTR mask = static_cast<DWORD_PTR>(kAllowedMask);
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        return static_cast<int>(GetLastError());
    }
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (unsigned i = 0; i < kCoreCount; ++i) {
        CPU_SET(i, &set);
    }
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        return errno;
    }
#endif
    // SetThreadAffinityMask / pthread_setaffinity_np succeed whenever the
    // requested mask *intersects* the process affinity, silently clamping to
    // the intersection. That would report success while running on fewer than
    // the required 0..3 cores, so verify the effective pin equals the mask.
    if (CurrentAllowedCores() != kAllowedMask) {
#if defined(_WIN32)
        return static_cast<int>(ERROR_ACCESS_DENIED);
#else
        return EPERM;
#endif
    }
    return 0;
}

std::uint32_t CpuAffinity::CurrentAllowedCores() noexcept {
#if defined(_WIN32)
    // TRACE-quality helper: SetThreadAffinityMask returns the thread's previous
    // affinity mask, so probing with all-ones reads the allowed set back. This
    // briefly migrates the thread onto the process's full core set (including
    // cores reserved for CUDA) before restoring, so it is not side-effect free.
    const DWORD_PTR probe = static_cast<DWORD_PTR>(-1);
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), probe);
    if (prev != 0) {
        SetThreadAffinityMask(GetCurrentThread(), prev);
    }
    return static_cast<std::uint32_t>(prev);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    std::uint32_t mask = 0;
    if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) == 0) {
        for (unsigned i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &set)) {
                mask |= (std::uint32_t(1) << i);
            }
        }
    }
    return mask;
#endif
}

}  // namespace lsxnetwork