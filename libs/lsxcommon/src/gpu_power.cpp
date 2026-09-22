#include "lsxcommon/gpu_power.h"

#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lsxcommon {
namespace perf {

namespace {

// Runtime-resolved NVML entry points. No nvml* symbol is referenced directly,
// so nothing links against libnvidia-ml / nvml.dll (or its CUDA stubs).
struct NvmlApi {
    void *handle = nullptr;
    nvmlReturn_t (*init)() = nullptr;
    nvmlReturn_t (*get_handle)(unsigned int, nvmlDevice_t *) = nullptr;
    nvmlReturn_t (*power_usage)(nvmlDevice_t, unsigned int *) = nullptr;
    nvmlReturn_t (*enforced_limit)(nvmlDevice_t, unsigned int *) = nullptr;
    nvmlDevice_t device = nullptr;
    bool ready = false;
};

void *LoadLibraryHandle()
{
#ifdef _WIN32
    if (HMODULE h = LoadLibraryA("nvml.dll")) return h;
    return LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
#else
    return dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
#endif
}

void *LoadSymbol(void *handle, const char *name)
{
    if (handle == nullptr) return nullptr;
#ifdef _WIN32
    return reinterpret_cast<void *>(
        GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

// Function-local static: initialized exactly once, thread-safe.
NvmlApi &Api()
{
    static NvmlApi api = [] {
        NvmlApi a;
        a.handle = LoadLibraryHandle();
        if (a.handle == nullptr) return a;
        a.init = reinterpret_cast<decltype(a.init)>(
            LoadSymbol(a.handle, "nvmlInit_v2"));
        a.get_handle = reinterpret_cast<decltype(a.get_handle)>(
            LoadSymbol(a.handle, "nvmlDeviceGetHandleByIndex_v2"));
        a.power_usage = reinterpret_cast<decltype(a.power_usage)>(
            LoadSymbol(a.handle, "nvmlDeviceGetPowerUsage"));
        a.enforced_limit = reinterpret_cast<decltype(a.enforced_limit)>(
            LoadSymbol(a.handle, "nvmlDeviceGetEnforcedPowerLimit"));
        if (a.init == nullptr || a.get_handle == nullptr ||
            a.power_usage == nullptr)
            return a;
        if (a.init() != NVML_SUCCESS) return a;
        if (a.get_handle(0, &a.device) != NVML_SUCCESS) return a;
        a.ready = true;
        return a;
    }();
    return api;
}

} // namespace

PowerStats ComputePowerStats(const std::vector<double> &samples_w)
{
    PowerStats s;
    if (samples_w.empty()) return s;
    double sum = 0.0;
    double peak = samples_w.front();
    for (double w : samples_w) {
        sum += w;
        peak = std::max(peak, w);
    }
    s.avg_w = sum / static_cast<double>(samples_w.size());
    s.peak_w = peak;
    s.samples = static_cast<int>(samples_w.size());
    s.valid = true;
    return s;
}

bool ReadGpuPowerWatts(double &out_w)
{
    NvmlApi &api = Api();
    if (!api.ready) return false;
    unsigned int milliwatts = 0;
    if (api.power_usage(api.device, &milliwatts) != NVML_SUCCESS) return false;
    out_w = static_cast<double>(milliwatts) / 1000.0;
    return true;
}

bool GpuPowerAvailable() { return Api().ready; }

double GpuPowerLimitW()
{
    NvmlApi &api = Api();
    if (!api.ready || api.enforced_limit == nullptr) return 0.0;
    unsigned int milliwatts = 0;
    if (api.enforced_limit(api.device, &milliwatts) != NVML_SUCCESS) return 0.0;
    return static_cast<double>(milliwatts) / 1000.0;
}

GpuPowerSampler::GpuPowerSampler(std::function<bool(double &)> reader,
                                 int interval_ms)
    : reader_(std::move(reader)),
      interval_ms_(interval_ms > 0 ? interval_ms : 20)
{
    if (!reader_) reader_ = ReadGpuPowerWatts;
}

GpuPowerSampler::~GpuPowerSampler()
{
    running_.store(false);
    if (thread_.joinable()) thread_.join();
}

void GpuPowerSampler::Start()
{
    if (running_.load()) return;
    if (thread_.joinable()) thread_.join();
    samples_.clear();
    running_.store(true);
    thread_ = std::thread([this] {
        while (running_.load()) {
            double watts = 0.0;
            if (!reader_(watts)) break;
            samples_.push_back(watts);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(interval_ms_));
        }
        running_.store(false);
    });
}

PowerStats GpuPowerSampler::Stop()
{
    running_.store(false);
    if (thread_.joinable()) thread_.join();
    return ComputePowerStats(samples_);
}

} // namespace perf
} // namespace lsxcommon
