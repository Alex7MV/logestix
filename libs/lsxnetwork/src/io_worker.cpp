#include "lsxnetwork/io_worker.h"
#include "lsxnetwork/affinity.h"

namespace lsxnetwork {

IoWorker::IoWorker(std::function<void()> loop) : loop_(std::move(loop)) {}

IoWorker::~IoWorker() noexcept {
    Stop();
}

int IoWorker::Start() {
    if (thread_.joinable()) {
        return -1;  // already started; reassigning a joinable std::thread terminates
    }
    running_.store(true, std::memory_order_release);
    pin_result_ = std::promise<int>{};
    auto future = pin_result_.get_future();
    thread_ = std::thread(&IoWorker::Entry, this);
    return future.get();
}

void IoWorker::Stop() noexcept {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void IoWorker::Entry() {
    const int rc = CpuAffinity::PinToCores0To3();
    pinned_.store(rc == 0, std::memory_order_release);
    pin_result_.set_value(rc);
    if (running_.load(std::memory_order_acquire)) {
        loop_();
    }
}

}  // namespace lsxnetwork