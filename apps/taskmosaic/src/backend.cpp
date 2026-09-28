#include "backend.h"

namespace taskmosaic {

Backend::Backend(const BackendConfig& cfg, int queue_depth)
    : cfg_(cfg), queue_depth_(queue_depth < 1 ? 1 : queue_depth) {}

int Backend::Connect() {
    const int rc = client_.Connect(
        cfg_.endpoint.c_str(),
        cfg_.cancel_endpoint.empty() ? nullptr : cfg_.cancel_endpoint.c_str());
    ready_.store(rc == 0, std::memory_order_release);
    return rc;
}

bool Backend::Probe(int timeout_ms) {
    if (!client_.Connected()) {
        ready_.store(false, std::memory_order_release);
        return false;
    }
    const bool ok = client_.Probe(timeout_ms);
    ready_.store(ok, std::memory_order_release);
    return ok;
}

bool Backend::TryAdmit() noexcept {
    int cur = admitted_.load(std::memory_order_relaxed);
    while (cur < queue_depth_) {
        if (admitted_.compare_exchange_weak(cur, cur + 1,
                                            std::memory_order_acq_rel))
            return true;
    }
    return false;
}

void Backend::Release() noexcept {
    admitted_.fetch_sub(1, std::memory_order_acq_rel);
}

}  // namespace taskmosaic
