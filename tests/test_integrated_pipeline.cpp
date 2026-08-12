#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/lsx_orchestrator.h>
#include <lsxnetwork/frame_queue.h>
#include <lsxnetwork/openai_endpoint.h>

#include <simdjson.h>

#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using lsxcommon::DispatchSignal;
using lsxcommon::LsxOrchestrator;
using lsxcommon::TaskMosaicStats;
using lsxcommon::ModelRequest;
using lsxcommon::SafetensorsModel;
using lsxcommon::ArrowTokenizerDb;
using lsxcommon::gpu::InferenceResult;
using lsxnetwork::FrameQueue;

namespace {

// Lightweight test double: records the number of Infer() dispatches per branch
// so the harness can assert exact routing without loading real weights.
class FakeModel final : public lsxcommon::IModel {
public:
    explicit FakeModel(std::string type) : type_(std::move(type)) {}

    bool Load() override { return frozen_ = true; }
    void Unload() override { frozen_ = false; }

    InferenceResult Infer(const ModelRequest&) override {
        ++inferred_;
        InferenceResult r;
        r.ok = true;
        r.num_tokens = 1;
        return r;
    }

    InferenceResult SelfCheck(const ModelRequest&) override {
        InferenceResult r;
        r.ok = true;
        return r;
    }

    SafetensorsModel Inspect() override { return {}; }
    const std::string& Type() const override { return type_; }
    ArrowTokenizerDb& Tokenizer() override { abort(); }

    std::uint64_t inferred_ = 0;

private:
    std::string type_;
    bool frozen_ = false;
};

std::string OpenAIRequestDoc(std::string_view model, std::int64_t prompt_tokens) {
    std::string doc;
    doc.reserve(384);
    doc += R"({"id":"chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAy0D","object":"chat.completion","model":")";
    doc += model;
    doc += R"(","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is Paris."},"finish_reason":"stop"}],"usage":{"prompt_tokens":)";
    doc += std::to_string(prompt_tokens);
    doc += R"(,"completion_tokens":8,"total_tokens":19}})";
    return doc;
}

}  // namespace

TEST_CASE("integrated: simultaneous GLM-5.2 + DeepSeek burst, zero token-drift",
          "[mosaic][integration][pipeline]")
{
    using lsxcommon::gpu::InferenceResult;

    FakeModel glm("glm-5.2");
    FakeModel deep("deepseek-v4-flash");
    LsxOrchestrator::ModelTable table = {
        {"glm-5.2", &glm},
        {"deepseek-v4-flash", &deep},
    };
    LsxOrchestrator orch(table);

    constexpr int kTotal = 4000;               // even: 2000 per branch
    constexpr int kPerBranch = kTotal / 2;

    FrameQueue queue(16u, 4096u);
    TaskMosaicStats stats;
    std::atomic<bool> stop{false};

    // Producer: concurrent burst of OpenAI-formatted requests, alternating
    // targets so both model branches are hammered simultaneously.
    std::thread producer([&queue, kTotal] {
        for (int i = 0; i < kTotal; ++i) {
            std::size_t cap = 0;
            char* chunk = nullptr;
            while ((chunk = queue.Acquire(&cap)) == nullptr) {
                std::this_thread::yield();
            }
            const std::string doc = OpenAIRequestDoc(
                (i % 2 == 0) ? "glm-5.2" : "deepseek-v4-flash",
                static_cast<std::int64_t>(i % 1000) + 1);
            REQUIRE(doc.size() <= cap);
            std::memcpy(chunk, doc.data(), doc.size());
            REQUIRE(queue.Publish(doc.size()) == 0);
        }
    });

    // Consumer: orchestration loop polls the lock-free tail watermark and
    // retires each slot only after the target branch has returned.
    std::thread consumer([&] { orch.ServeTaskMosaic(queue, stats, stop); });

    producer.join();

    // Drain barrier: only set stop once every published frame has been taken
    // off the ring, so the consumer retires the full burst (no stranded slots).
    while (stats.taken_frames.load() < static_cast<std::uint64_t>(kTotal)) {
        std::this_thread::yield();
    }
    stop.store(true, std::memory_order_release);
    consumer.join();

    // 0 token-drift: every frame taken off the ring was retired exactly once.
    REQUIRE(stats.taken_frames == static_cast<std::uint64_t>(kTotal));
    REQUIRE(stats.retired_slots == static_cast<std::uint64_t>(kTotal));
    CHECK(stats.glm_routed == static_cast<std::uint64_t>(kPerBranch));
    CHECK(stats.deepseek_routed == static_cast<std::uint64_t>(kPerBranch));
    CHECK(stats.unknown_routed == 0u);
    CHECK(queue.Empty());   // ring fully drained, slots reusable

    // Both execution pipelines were invoked through their own branch.
    CHECK(glm.inferred_ == static_cast<std::uint64_t>(kPerBranch));
    CHECK(deep.inferred_ == static_cast<std::uint64_t>(kPerBranch));
}

TEST_CASE("integrated: unknown model string is retired (not leaked) and routed",
          "[mosaic][integration][pipeline]")
{
    FakeModel glm("glm-5.2");
    LsxOrchestrator::ModelTable table = {{"glm-5.2", &glm}};
    LsxOrchestrator orch(table);

    FrameQueue queue(4u, 4096u);
    TaskMosaicStats stats;
    std::atomic<bool> stop{false};

    std::size_t cap = 0;
    char* chunk = queue.Acquire(&cap);
    const std::string doc = OpenAIRequestDoc("llama-9000", 5);
    REQUIRE(doc.size() <= cap);
    std::memcpy(chunk, doc.data(), doc.size());
    REQUIRE(queue.Publish(doc.size()) == 0);

    std::thread consumer([&] { orch.ServeTaskMosaic(queue, stats, stop); });
    // Let the consumer drain the single frame.
    while (stats.retired_slots.load() == 0u) {
        std::this_thread::yield();
    }
    stop.store(true, std::memory_order_release);
    consumer.join();

    CHECK(stats.taken_frames == 1u);
    CHECK(stats.retired_slots == 1u);   // unknown frame still retired -> no leak
    CHECK(stats.unknown_routed == 1u);
    CHECK(stats.glm_routed == 0u);
    CHECK(stats.deepseek_routed == 0u);
    CHECK(queue.Empty());
}

TEST_CASE("integrated: Classify maps both targets + unknown", "[mosaic][integration]")
{
    CHECK(LsxOrchestrator::Classify("glm-5.2") == DispatchSignal::kGlm52);
    CHECK(LsxOrchestrator::Classify("deepseek-v4-flash") == DispatchSignal::kDeepSeekFlash);
    CHECK(LsxOrchestrator::Classify("") == DispatchSignal::kUnknown);
    CHECK(LsxOrchestrator::Classify("other") == DispatchSignal::kUnknown);
}

TEST_CASE("integrated: fixture doc parses and exposes the model field",
          "[mosaic][integration][parse]")
{
    using lsxnetwork::OpenAiEndpoint;
    using lsxnetwork::ChatCompletion;

    const std::string doc = OpenAIRequestDoc("deepseek-v4-flash", 42);
    INFO("doc bytes=[" << doc << "]");
    std::vector<char> buf(doc.size() + FrameQueue::kSimdjsonPadding, 0);
    std::memcpy(buf.data(), doc.data(), doc.size());

    ChatCompletion cc;
    const simdjson::error_code rc =
        OpenAiEndpoint().Parse(buf.data(), doc.size(), buf.size(), cc);
    REQUIRE(rc == simdjson::SUCCESS);
    CHECK(cc.id == "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAy0D");
    CHECK(cc.object == "chat.completion");
    CHECK(cc.model == "deepseek-v4-flash");
    CHECK(cc.content == "The capital of France is Paris.");
    CHECK(cc.prompt_tokens == 42);
}
