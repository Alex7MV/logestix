#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

namespace lsxcommon {

// Flat, host-only streaming token packet. Transmitted by const reference so no
// allocation or copy occurs on the model->caller hot path.
struct TokenChunk {
    std::uint32_t token_id = 0;
    std::uint32_t step_t = 0;
    float logit_maxv = 0.0f;
    bool is_special = false;
};

using TokenCallback = std::function<void(const TokenChunk&)>;

// CUDA-safe streaming handle threaded into every generation loop. Members are
// non-owning pointers; both may be null (streaming disabled).
struct GenerationStream {
    const TokenCallback* on_token = nullptr;
    const std::atomic<bool>* abort_flag = nullptr;

    bool Aborted() const {
        return abort_flag != nullptr &&
               abort_flag->load(std::memory_order_acquire);
    }

    void Emit(std::uint32_t token_id, std::uint32_t step) const {
        if (on_token == nullptr || !*on_token) return;
        TokenChunk c;
        c.token_id = token_id;
        c.step_t = step;
        (*on_token)(c);
    }
};

}  // namespace lsxcommon
