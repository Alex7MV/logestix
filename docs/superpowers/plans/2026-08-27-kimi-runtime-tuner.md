# Kimi RuntimeTuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task (субагенты в этом репозитории ЗАПРЕЩЕНЫ —
> AGENTS.md). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести Kimi-K2.7-Code до полного GLM-паритета адаптивного expert-cache:
телеметрия hits/misses из CUDA-пайплайна + `lifecycle::RuntimeTuner` между
запросами + governor lease на кэш-резиденцию.

**Architecture:** Три слоя. (1) Хост-хелперы в `kimi_moe_expert_cache.h` —
чистые inline-функции `KimiCountCacheHits`/`KimiExpertCachePerExpertBytes`
(unit-тест без CUDA). (2) CUDA-слой — поля телеметрии в `KimiGenerationResult`,
накопление на сайте `KimiMoeResolveBatch`, атомарный override слотов
`SetKimiExpertCacheSlotsOverride` + realloc арены при расхождении (паритет
`GlmExpertCacheSlotsOverride`, glm_moe_inference.cu:29). (3) Модель —
`RuntimeTuner` в `KimiK27CodeModel`, lease через `ModelBase::
AcquireExpertCacheLease` (уже существует), `Report()` между запросами.

**Tech Stack:** C++20, CUDA 13.3 (Blackwell sm_120/121), Catch2 3.7.0,
ResourceGovernor (leases), lifecycle::RuntimeTuner.

**Инварианты:**
- Бит-точность: телеметрия — только счётчики; realloc арены меняет число
  слотов, но слот читает ТЕ же упакованные байты теми же ядрами
  (контракт kimi_moe_expert_cache.h:13).
- TDD: каждый шаг RED → прогон → GREEN → прогон → commit.
- Windows-ограничение (AGENTS.md): исполнять ТОЛЬКО host-only тесты
  (`test_kimi_moe_expert_cache`, `test_model_runtime`, ...); Kimi GPU-тесты
  (`test_kimi_moe_*`) — только СБОРКА на Windows, прогон на Linux
  (Ubuntu 26.04, `ctest --preset linux-debug -R kimi`).

---

### Task 1: Хост-хелперы телеметрии (`KimiCountCacheHits`, `KimiExpertCachePerExpertBytes`)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_expert_cache.h`
- Test: `tests/test_kimi_moe_expert_cache.cpp`

- [ ] **Step 1: Write the failing test**

В `tests/test_kimi_moe_expert_cache.cpp` после существующих test cases добавить:

```cpp
TEST_CASE("KimiCountCacheHits accounts residents vs misses", "[kimi][cache]")
{
    using lsxcommon::kimi::KimiCountCacheHits;
    using lsxcommon::kimi::KimiMoeResolveBatch;
    using lsxcommon::kimi::KimiMoeExpertCachePlan;
    constexpr int kEPL = 4;
    KimiMoeExpertCachePlan p;
    p.Init(2, 2 * kEPL);   // 2 slots, 8 экспертов

    // Первый батч: всё miss (кэш пуст) -> 0 hits / 8 misses.
    const int ids0[8] = {0, 1, 2, 3, 0, 1, 2, 3};
    KimiMoeCacheBatch b0 = KimiMoeResolveBatch(p, 0, kEPL, ids0, 8);
    auto [h0, m0] = KimiCountCacheHits(b0, 8);
    CHECK(h0 == 0);
    CHECK(m0 == 8);

    // Повтор: все resident -> 8 hits / 0 misses.
    KimiMoeCacheBatch b1 = KimiMoeResolveBatch(p, 0, kEPL, ids0, 8);
    auto [h1, m1] = KimiCountCacheHits(b1, 8);
    CHECK(h1 == 8);
    CHECK(m1 == 0);

    // Смешанный: 4 resident + 4 eviction-miss.
    const int ids2[8] = {4, 5, 6, 7, 0, 1, 2, 3};
    KimiMoeCacheBatch b2 = KimiMoeResolveBatch(p, 1, kEPL, ids2, 8);
    auto [h2, m2] = KimiCountCacheHits(b2, 8);
    CHECK(h2 == 4);
    CHECK(m2 == 4);

    // Неактивный план: k токенов, все miss.
    KimiMoeExpertCachePlan off;
    off.Init(0, 2 * kEPL);
    KimiMoeCacheBatch b3 = KimiMoeResolveBatch(off, 0, kEPL, ids0, 8);
    auto [h3, m3] = KimiCountCacheHits(b3, 8);
    CHECK(h3 == 0);
    CHECK(m3 == 8);
}

TEST_CASE("KimiExpertCachePerExpertBytes matches the arena stride math",
          "[kimi][cache]")
{
    // 3 матрицы pack (fp16-пары) + 3 блока scale (fp8/16 байт на элемент).
    const std::size_t expect =
        3ull * (7168ull * 2048ull / 2) +      // packed w1|w3|w2
        3ull * (7168ull * 2048ull / 16);      // scales s1|s3|s2
    CHECK(lsxcommon::kimi::KimiExpertCachePerExpertBytes() == expect);
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target test_kimi_moe_expert_cache
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build -C Debug -R test_kimi_moe_expert_cache --output-on-failure
```

Expected: compile FAIL — `KimiCountCacheHits`/`KimiExpertCachePerExpertBytes`
не являются членами `lsxcommon::kimi` (error C2039).

- [ ] **Step 3: Write minimal implementation**

В `libs/lsxcommon/include/lsxcommon/kimi_moe_expert_cache.h` после
`KimiMoeResolveBatch` (после строки ~167) добавить:

```cpp
// Telemetry (RuntimeTuner): residents get resident_slot >= 0 at resolve
// time; everything else in the batch is a miss (upload or hard-miss).
// Returns {hits, misses} for one resolved top-k batch.
inline std::pair<std::int64_t, std::int64_t>
KimiCountCacheHits(const KimiMoeCacheBatch &b, int k)
{
    std::int64_t hits = 0;
    for (int i = 0; i < k; ++i)
        if (b.resident_slot[i] >= 0) ++hits;
    const std::int64_t misses =
        static_cast<std::int64_t>(b.missing_count);
    return {hits, misses};
}

// Worst-case per-expert device footprint of one cache slot: packed
// [w1|w3|w2] fp16-pairs + scales [s1|s3|s2] (16-element E8M0 blocks).
// Single authority shared by the model Load lease/tuner sizing and the
// lazy arena allocation in kimi_moe_inference.cu (was a hardcoded
// expression there; kKimiExpertMatEl = 7168*2048).
inline std::size_t KimiExpertCachePerExpertBytes()
{
    constexpr std::size_t kMatEl = 7168ull * 2048ull;
    return 3ull * (kMatEl / 2) + 3ull * (kMatEl / 16);
}
```

(`#include <cstdint>` уже есть; добавить `#include <utility>` для
`std::pair`, если отсутствует.)

- [ ] **Step 4: Run test to verify it passes**

Команда из Step 2. Expected: PASS, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/kimi_moe_expert_cache.h tests/test_kimi_moe_expert_cache.cpp
git commit -m "feat: kimi cache telemetry host helpers (hits/misses, per-expert bytes)"
```

---

### Task 2: Override слотов + realloc арены в CUDA-слое (GLM-паритет)

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_kernels.h`
- Modify: `libs/lsxcommon/src/kimi_moe_inference.cu` (lazy-блок ~строки 1170-1205,
  resolve-сайт ~строки 1015-1055)

**Примечание:** на Windows только компиляция; прогон — Linux.

- [ ] **Step 1: Write the failing test (compile-level RED)**

Хост-часть RED уже покрыта Task 1; для CUDA-слоя RED — это компиляция нового
символа `SetKimiExpertCacheSlotsOverride`, на который ссылается Task 4.
Пропустить прогон, зафиксировать ожидание: до реализации линковка Task 4
упадёт с LNK2019. Идти сразу к Step 2.

- [ ] **Step 2: Override-атомика в `kimi_moe_kernels.h`**

Рядом с `RunKimiGeneration` (после строки 75) добавить:

```cpp
// RuntimeTuner hook (GLM parity, glm_moe_inference.cu:29): slots >= 0
// forces the lazy arena to that slot count; -1 falls back to the env
// heuristic. Called between inference requests, never in the hot loop.
void SetKimiExpertCacheSlotsOverride(int slots);
int KimiExpertCacheSlotsOverride();
```

- [ ] **Step 3: Реализация в `kimi_moe_inference.cu`**

Вверху файла (рядом с другими file-scope статиками) добавить:

```cpp
std::atomic<int> g_kimi_cache_slots_override{-1};

void SetKimiExpertCacheSlotsOverride(int slots)
{
    g_kimi_cache_slots_override.store(slots, std::memory_order_relaxed);
}

int KimiExpertCacheSlotsOverride()
{
    return g_kimi_cache_slots_override.load(std::memory_order_relaxed);
}
```

Ленивый блок аллокации (~строка 1170) заменить на: override → env →
эвристика, БЕЗ вычисления per_expert вручную (Task 1-хелпер), и с realloc
при расхождении числа слотов:

```cpp
    if (cache && !cache->device) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            const int total_experts = kKimiNumLayers * kKimiNumRouted;
            const size_t per_expert = KimiExpertCachePerExpertBytes();
            int want = KimiExpertCacheSlotsOverride();
            if (want < 0)
                want = KimiMoeExpertCacheSlots(
                    free_b, per_expert, total_experts,
                    std::getenv("LSX_KIMI_EXPERT_CACHE_SLOTS"));
            if (want > total_experts) want = total_experts;
            if (want < 0) want = 0;
            cache->plan.Init(want, total_experts);
            std::string cerr_;
            int try_slots = want;
            while (try_slots >= 1 && !cache->device) {
                if (AllocateKimiMoeExpertCache(db, try_slots, cache->device,
                                               cerr_) == cudaSuccess)
                    break;
                try_slots /= 2;
            }
            if (cache->device)
                log::info("[kimi-infer] expert cache: free=" +
                          std::to_string(free_b >> 20) + "MiB per_expert=" +
                          std::to_string(per_expert >> 20) + "MiB total=" +
                          std::to_string(total_experts) + " slots=" +
                          std::to_string(cache->device->slots));
            else if (want > 0)
                log::warn("[kimi-infer] expert cache unavailable (" + cerr_ +
                          "); legacy staging path");
        }
    } else if (cache && cache->device) {
        // RuntimeTuner resized: rebuild the arena when the wanted slot
        // count no longer matches. Same packed bytes per slot -> bit-exact.
        const int want = KimiExpertCacheSlotsOverride();
        if (want >= 0 && want != cache->device->slots) {
            FreeKimiMoeExpertDeviceCache(cache->device);
            cache->device = nullptr;
            std::string cerr_;
            int try_slots = want;
            const int total_experts = cache->plan.total_experts;
            while (try_slots >= 1 && !cache->device) {
                if (AllocateKimiMoeExpertCache(db, try_slots, cache->device,
                                               cerr_) == cudaSuccess)
                    break;
                try_slots /= 2;
            }
            cache->plan.Init(cache->device ? cache->device->slots : 0,
                             total_experts);
            if (cache->device)
                log::info("[kimi-infer] expert cache resized to " +
                          std::to_string(cache->device->slots) + " slots");
            else
                log::warn("[kimi-infer] expert cache rebuild failed (" +
                          cerr_ + "); legacy staging path");
        }
    }
```

В `KimiGenerationResult`-накопление (resolve-сайт ~строка 1018) добавить
после `KimiMoeCacheBatch b = KimiMoeResolveBatch(...)`:

```cpp
                    const auto [b_hits, b_misses] =
                        KimiCountCacheHits(b, kKimiTopK);
                    batch_hits += b_hits;
                    batch_misses += b_misses;
```

(переменные `batch_hits`/`batch_misses` — локальные `std::int64_t = 0;`
рядом с объявлением счётчиков шага; structured bindings требуют C++20 —
включено). В конце `RunKimiGeneration` перед возвратом успеха опубликовать:

```cpp
    gen.cache_hits = batch_hits;
    gen.cache_misses = batch_misses;
```

- [ ] **Step 4: Verify build**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxcommon
```

Expected: компиляция без ошибок.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/kimi_moe_kernels.h libs/lsxcommon/src/kimi_moe_inference.cu
git commit -m "feat: kimi expert-cache slots override + arena rebuild + hits/misses telemetry"
```

---

### Task 3: Телеметрия в `KimiGenerationResult`

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_kernels.h:55-65` (struct
  `KimiGenerationResult`)

Входит в Task 2 по факту (поля нужны для компиляции накопления), но
зафиксировано отдельным шагом:

- [ ] **Step 1: Добавить поля**

```cpp
struct KimiGenerationResult {
    std::vector<int> tokens;      // generated tokens only (never the prompt)
    float first_maxv = 0.0f;
    bool ok = false;
    bool eos_terminated = false;
    int kv_slots_used = 0;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    std::string error_msg;
    std::vector<float> final_logits;
    // Expert-cache telemetry since generation start (RuntimeTuner input;
    // GLM parity: glm_moe_inference.h:404). Zero when the cache is off.
    std::int64_t cache_hits = 0;
    std::int64_t cache_misses = 0;
};
```

- [ ] **Step 2: Build + commit**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxcommon
git add libs/lsxcommon/src/kimi_moe_kernels.h
git commit -m "feat: kimi generation result carries expert-cache telemetry"
```

---

### Task 4: RuntimeTuner в `KimiK27CodeModel`

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_model.h`
- Modify: `libs/lsxcommon/src/kimi_moe_model.cpp` (Load ~строки 60-105,
  Infer ~строки 105-135, TeardownWeights)

- [ ] **Step 1: Члены модели (header)**

В private-секцию `KimiK27CodeModel` (после `expert_cache_`) добавить
include `"lsxcommon/runtime_tuner.h"` в начало файла и члены:

```cpp
    // Adaptive expert-cache controller (GLM parity): owns the cache lease
    // (moved from ModelBase::expert_cache_lease_) and resizes between
    // requests from hits/misses telemetry.
    std::unique_ptr<lifecycle::RuntimeTuner> tuner_;
    std::int64_t tuner_last_hits_ = 0;
    std::int64_t tuner_last_misses_ = 0;
```

- [ ] **Step 2: Load — lease + tuner после `AcquirePermanentWeights`**

В `KimiK27CodeModel::Load` после строки
`if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();`
вставить:

```cpp
    // Adaptive expert-cache lease + RuntimeTuner (GLM parity). Bounds span
    // the whole slot range: the tuner grows/shrinks from telemetry.
    const int total_experts = kimi::kKimiNumLayers * kimi::kKimiNumRouted;
    const std::uint64_t per_expert = kimi::KimiExpertCachePerExpertBytes();
    lifecycle::TuningBounds bounds;
    bounds.expert_cache_slots_min = 4;
    bounds.expert_cache_slots_max = static_cast<std::size_t>(total_experts);
    int initial_slots = 0;
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
            initial_slots = kimi::KimiMoeExpertCacheSlots(
                free_b, per_expert, total_experts,
                std::getenv("LSX_KIMI_EXPERT_CACHE_SLOTS"));
    }
    if (initial_slots > 0) {
        AcquireExpertCacheLease(
            static_cast<std::uint64_t>(initial_slots) * per_expert,
            bounds.expert_cache_slots_min * per_expert,
            bounds.expert_cache_slots_max * per_expert);
        auto &gov = ResourceGovernor::Instance();
        tuner_ = std::make_unique<lifecycle::RuntimeTuner>(
            &gov, std::move(expert_cache_lease_), per_expert, bounds);
        kimi::SetKimiExpertCacheSlotsOverride(
            static_cast<int>(tuner_->Slots()));
        log::info("[kimi-model] expert-cache tuner: initial slots=" +
                  std::to_string(tuner_->Slots()));
    } else {
        log::info("[kimi-model] expert cache disabled (env/heuristic); "
                  "legacy staging path");
    }
    tuner_last_hits_ = tuner_last_misses_ = 0;
```

(удалить прежнюю строку `expert_cache_.plan.Init(0, ...)` в Load — план
инициализирует ленивый блок .cu; `#include <cstdlib>` для `std::getenv`
уже есть в TU через другие заголовки — при отсутствии добавить.)

- [ ] **Step 3: Infer — Report между запросами**

В `KimiK27CodeModel::Infer` после успешного `RunUnderResidencyLock` и
перед `runtime::FillInferenceResult(...)`:

```cpp
    if (tuner_) {  // between requests: adapt the expert cache (GLM parity)
        tuner_->Report(gen.cache_hits - tuner_last_hits_,
                       gen.cache_misses - tuner_last_misses_);
        tuner_last_hits_ = gen.cache_hits;
        tuner_last_misses_ = gen.cache_misses;
        kimi::SetKimiExpertCacheSlotsOverride(
            static_cast<int>(tuner_->Slots()));
        ResourceGovernor::Instance().ReconcileVram();
    }
```

- [ ] **Step 4: TeardownWeights — tuner первым**

В начале `KimiK27CodeModel::TeardownWeights` добавить `tuner_.reset();`
(lease отпускается деструктором тюнера; moved-from
`ModelBase::expert_cache_lease_` пуст — повторный release в базовом
Unload безопасен).

- [ ] **Step 5: Verify build + host-only suites**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build -C Debug -R "test_kimi_moe_expert_cache|test_model_runtime|test_resource_governor|test_runtime_tuner" --output-on-failure
```

Expected: сборка без ошибок; все 4 цели PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/kimi_moe_model.h libs/lsxcommon/src/kimi_moe_model.cpp
git commit -m "feat: kimi model RuntimeTuner wiring (lease, telemetry report, slots override)"
```

---

### Task 5: Верификация (Linux + Windows) и финализация

- [ ] **Step 1: Windows — полная сборка + host-only регресс**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build -C Debug -R "test_kimi_moe_expert_cache|test_model_runtime|test_model_engine|test_model_factory|test_geometry_guard|test_resource_governor|test_runtime_tuner|test_lifecycle_plan" --output-on-failure
```

Expected: 0 failed. Kimi GPU-цели только СБОРАТСЯ (ограничение AGENTS.md).

- [ ] **Step 2: Linux (Ubuntu 26.04) — прогон Kimi GPU-тестов**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug -R kimi --output-on-failure
```

Expected: `test_kimi_moe_expert_cache`, `test_kimi_moe_convert`,
`test_kimi_moe_model`, `test_kimi_moe_inference`, `test_kimi_moe_selfcheck`
— PASS/SKIP (без чекпоинта SKIP_RETURN_CODE 4). Реальная модель обязана
генерировать "Paris" как раньше (бит-точность не затронута).

- [ ] **Step 3: Финальный коммит (если остались правки)**

```bash
git add -A
git commit -m "test: kimi runtime tuner verification pass"
git push
```

---

## Self-Review

- **Покрытие цели:** телеметрия (Task 1+3), CUDA-накопление + override/realloc
  (Task 2), модельный tuner + lease (Task 4), верификация (Task 5) — цель
  закрыта полностью.
- **Плейсхолдеры:** нет; все шаги содержат код/команды.
- **Типы:** `KimiCountCacheHits` возвращает `std::pair<std::int64_t,
  std::int64_t>` — используется structured bindings в Task 2 и test в Task 1;
  `RuntimeTuner(gov, lease, stride, bounds)` — сигнатура
  `runtime_tuner.h:31`; `AcquireExpertCacheLease`/`expert_cache_lease_` —
  `model_base.h` (lifecycle-unification II, коммит c2f6f72).
