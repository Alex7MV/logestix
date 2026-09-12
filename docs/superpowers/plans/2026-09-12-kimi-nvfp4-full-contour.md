# Kimi-K2.7-Code — полный NVFP4-контур + максимальная скорость

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Subagents ЗАПРЕЩЕНЫ политикой AGENTS.md — выполнение только inline основной
> сессией. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Конвертировать Kimi-K2.7-Code в полный NVFP4-контур (конвертер/репак/лоадер), замерить decode t/s, довести до максимума оборудования RTX 5090 32GB, найти и исправить причины деградации.

**Architecture:** Kimi уже на формате `logestix-kimi-r2` (MLA NVFP4 + routed-эксперты NVFP4 через `PackExpertNVFP4`, раунд-2, 127/127). Эксперты — NVFP4 E2M1+E4M3/16 (24.77 MiB/эксперт × 60×384 = 23040 → ~558 GiB vault). Decode остался на **1.3 t/s** (раунд-2): файловый mmap-источник H2D ~17 GiB/s effective. Решение — перенос доказанного DeepSeek-пакета: **pinned expert store** (один cudaHostAlloc-пул, layout [w1|w3|w2][s1|s3|s2], LSX_EXPERT_PIN=0 opt-out) поверх уже работающих full-resident mmaps (max_open=65536 дефолт стора) + LRU-кэша из свободной VRAM. Загрузка в кэш-слоты (6×cudaMemcpyAsync из entry-поинтеров) не меняется — pinned-источник ускоряет её автоматически.

**Tech Stack:** CUDA 13.4 / Blackwell sm_120, CMake presets linux-release-znver4, Catch2, spdlog, abseil (host-only).

## Global Constraints

- CPU/GPU-эксклюзив: lsxbenchmark/perf-тесты не запускать параллельно с другими GPU-нагрузками (AGENTS.md).
- RAM: 750 GiB total; pin ~558 GiB допустим (политика «RAM не экономим»), fallback на mmap при неудаче cudaHostAlloc — обязан быть.
- Bit-exact SelfCheck "Paris" ×3 (LSX_TEST_KIMI_SELFCHECK_CYCLES / SKIP_RETURN_CODE 4).
- Тесты Windows-only env: только `setenv` под `#ifdef _WIN32` guard (эталон `test_qwen_nvfp4_gemv.cu`).
- Все логи — spdlog; новые тесты — label в том же коммите.
- Физический потолок: miss-трафик ~7–12 GiB/токен; pinned H2D ~45–50 GiB/s (DeepSeek-замер) ⇒ потолок ~4–6 t/s при нулевом reuse, выше при hit-rate. Ниже 80% потолка — искать причину (nsys).

---

### Task 1: Конвертация vault + baseline-замер (ops, GPU-эксклюзив)

**Files:**
- Create: `/mnt/llm_models/models/Kimi-K2.7-Code/logestix/` (vault, вне репо)
- Create: `/tmp/opencode/kimi_convert.log`

**Interfaces:**
- Consumes: `lsxforge --kimi --input <dir> --output <dir>` (main.cpp:324 → `kimi::ConvertKimiModel`), in-place как DeepSeek.
- Produces: vault `logestix/model.index.json` (формат `logestix-kimi-r2`, `mla_nvfp4`), `logestix/pinned/*.arrow`, `logestix/experts/layer_N/exp_E.arrow`; baseline t/s в журнале задачи.

- [ ] **Step 1: Пересобрать текущий main (fresh lsxforge)**

```bash
cmake --build build/linux-release-znver4 -j 160 --target lsxforge 2>&1 | tail -2
```
Expected: `ninja: no work to do` либо успешная линковка, 0 errors.

- [ ] **Step 2: Запустить конвертацию in-place (фон, 20–60 мин)**

```bash
nohup ./build/linux-release-znver4/apps/lsxforge/lsxforge --kimi \
  --input /mnt/llm_models/models/Kimi-K2.7-Code \
  --output /mnt/llm_models/models/Kimi-K2.7-Code \
  > /tmp/opencode/kimi_convert.log 2>&1 &
echo $! > /tmp/opencode/kimi_convert.pid
```
Expected: PID записан; лог растёт. Пока идёт — GPU занят, никаких CUDA-запусков.

- [ ] **Step 3: Дождаться завершения, проверить vault**

```bash
while kill -0 $(cat /tmp/opencode/kimi_convert.pid) 2>/dev/null; do sleep 60; done
tail -3 /tmp/opencode/kimi_convert.log
python3 -c "
import json;d=json.load(open('/mnt/llm_models/models/Kimi-K2.7-Code/logestix/model.index.json'))
print(d['format'], d['layers'], d['experts_per_layer'], d.get('total_experts'), 'mla' , 'mla_nvfp4' in d)"
ls /mnt/llm_models/models/Kimi-K2.7-Code/logestix/experts | wc -l
```
Expected: лог `Done ... experts processed, incomplete 0 / Conversion complete`; формат `logestix-kimi-r2`; experts dir = 60 слоёв (layer_1..layer_60; layer_0 dense); `mla_nvfp4` присутствует.

- [ ] **Step 4: Baseline-замер (GPU эксклюзивно)**

```bash
LSX_TEST_KIMI_MODEL_PATH=/mnt/llm_models/models/Kimi-K2.7-Code \
  timeout 600 ./build/linux-release-znver4/tests/test_blackwell_perf_kimi 2>&1 | grep -E "PERF|tok/s|expert cache|pinned"
```
Expected: tok/s ≈ 1.3 (раунд-2 воспроизведение); запись числа в журналы задачи как baseline.

---

### Task 2: Pinned expert store для Kimi (TDD)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_load.h:148-164` (KimiExpertEntry +3 поля)
- Modify: `libs/lsxcommon/src/kimi_moe_load.cpp` (`load_kimi_expert_index`, pin-проход)
- Test: `tests/test_kimi_moe_loader.cpp` (расширить "Kimi expert index resolves routed experts" + новый negative-case)

**Interfaces:**
- Consumes: `ExpertStore::pinned_pool/pinned_bytes` (уже в expert_store.h, добавлены в DeepSeek round-8), `Free()` уже делает `cudaFreeHost(pinned_pool)`.
- Produces: `KimiExpertEntry::pinned_blob (const uint8_t*), blob_pack_bytes (int64_t), blob_scale_bytes (int64_t)`; pin-проход внутри `load_kimi_expert_index`; env `LSX_EXPERT_PIN=0` отключает.

- [ ] **Step 1: Красный тест — pinned-пул в синтетическом fixture**

В `tests/test_kimi_moe_loader.cpp` расширить `TEST_CASE("Kimi expert index resolves routed experts", ...)`:

```cpp
    // Pinned store (r3): blobs live in one locked host pool; pointers stay
    // valid for async H2D without the file-backed VMA penalty.
    REQUIRE(db.pinned_pool != nullptr);
    REQUIRE(db.pinned_bytes >= static_cast<std::size_t>(e.blob_pack_bytes
                                                     + e.blob_scale_bytes));
    REQUIRE(e.pinned_blob != nullptr);
    REQUIRE(e.pinned_blob >= static_cast<const uint8_t *>(db.pinned_pool));
    REQUIRE(e.pinned_blob < static_cast<const uint8_t *>(db.pinned_pool)
                              + db.pinned_bytes);
    // pair layout [w1|w3|w2][s1|s3|s2] — DeepSeek round-8 convention
    REQUIRE(e.blob_pack_bytes == e.w1_bytes + e.w3_bytes + e.w2_bytes);
    REQUIRE(e.blob_scale_bytes == e.w1_scales_bytes + e.w3_scales_bytes
                                + e.w2_scales_bytes);
    // upload source pointers were re-pointed INTO the pool
    REQUIRE(e.host_w1 >= static_cast<const uint8_t *>(db.pinned_pool));
    REQUIRE(e.host_w1 < static_cast<const uint8_t *>(db.pinned_pool)
                           + db.pinned_bytes);
```

- [ ] **Step 2: Запустить тест — красный**

```bash
cmake --build build/linux-release-znver4 -j 160 --target test_kimi_moe_loader 2>&1 | tail -1
LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4/tests/test_kimi_moe_loader "Kimi expert index resolves routed experts" 2>&1 | tail -5
```
Expected: FAIL — `pinned_pool` ещё не заполняется (поле, вероятно, уже есть в сторе — тогда nullptr).

- [ ] **Step 3: Реализация — поля entry**

`kimi_moe_load.h`, в `struct KimiExpertEntry` после `w3_scales_bytes`:

```cpp
    // Pinned-store view (DeepSeek round-8 parity): when the whole store is
    // pinned into locked host memory, blobs live contiguously per expert as
    // packed [w1|w3|w2] then scales [s1|s3|s2]; host_* pointers are re-pointed
    // into the pool and pinned_blob marks the expert block start.
    const uint8_t *pinned_blob = nullptr;
    int64_t blob_pack_bytes = 0;
    int64_t blob_scale_bytes = 0;
```

- [ ] **Step 4: Реализация — pin-проход в `load_kimi_expert_index`**

После `db.Load(model_path, /*fallback=*/true);` вставить (mirror deepseek_moe_load.cu:280-355, но per-entry tolerant — dense layer 0 не имеет экспертов):

```cpp
    // ── Pin the routed-expert store into locked host memory ──────────────
    // File-backed mmap sources carry a driver VMA-lookup penalty on every
    // H2D copy and vanish under munmap while an async copy is in flight.
    // One cudaHostAlloc pool, per expert packed [w1|w3|w2] then scales
    // [s1|s3|s2] — the order the slot upload streams. Entries without
    // weights (dense layer 0 of the 61-layer passport) are skipped.
    // Disable with LSX_EXPERT_PIN=0 (falls back to mmap views).
    if (std::getenv("LSX_EXPERT_PIN") == nullptr
        || std::getenv("LSX_EXPERT_PIN")[0] != '0') {
        int64_t pool_bytes = 0;
        for (const auto &en : db.entries) {
            if (!en.host_w1 || !en.host_w2 || !en.host_w3
                || !en.host_w1_scales || !en.host_w2_scales
                || !en.host_w3_scales)
                continue;   // dense layer or partial expert
            pool_bytes += ((en.w1_bytes + en.w3_bytes + en.w2_bytes + 255)
                           / 256) * 256;
            pool_bytes += ((en.w1_scales_bytes + en.w3_scales_bytes
                            + en.w2_scales_bytes + 255) / 256) * 256;
        }
        void *pool = nullptr;
        if (pool_bytes > 0
            && cudaHostAlloc(&pool, static_cast<size_t>(pool_bytes),
                             cudaHostAllocDefault) == cudaSuccess) {
            db.pinned_pool = pool;
            db.pinned_bytes = static_cast<std::size_t>(pool_bytes);
            uintptr_t cur = reinterpret_cast<uintptr_t>(pool);
            auto aln = [](uintptr_t v) { return (v + 255) / 256 * 256; };
            int pinned_cnt = 0;
            for (auto &en : db.entries) {
                if (!en.host_w1 || !en.host_w2 || !en.host_w3
                    || !en.host_w1_scales || !en.host_w2_scales
                    || !en.host_w3_scales)
                    continue;
                en.pinned_blob = reinterpret_cast<const uint8_t *>(cur);
                en.blob_pack_bytes = en.w1_bytes + en.w3_bytes + en.w2_bytes;
                en.blob_scale_bytes = en.w1_scales_bytes + en.w3_scales_bytes
                                    + en.w2_scales_bytes;
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w1,
                            static_cast<std::size_t>(en.w1_bytes));
                en.host_w1 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w1_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w3,
                            static_cast<std::size_t>(en.w3_bytes));
                en.host_w3 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w3_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w2,
                            static_cast<std::size_t>(en.w2_bytes));
                en.host_w2 = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w2_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w1_scales,
                            static_cast<std::size_t>(en.w1_scales_bytes));
                en.host_w1_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w1_scales_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w3_scales,
                            static_cast<std::size_t>(en.w3_scales_bytes));
                en.host_w3_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w3_scales_bytes);
                std::memcpy(reinterpret_cast<void *>(cur), en.host_w2_scales,
                            static_cast<std::size_t>(en.w2_scales_bytes));
                en.host_w2_scales = reinterpret_cast<const uint8_t *>(cur);
                cur = aln(cur + en.w2_scales_bytes);
                ++pinned_cnt;
            }
            log::info("[kimi_load] pinned routed-expert store: "
                      + std::to_string(pinned_cnt) + " experts, "
                      + std::to_string(pool_bytes / (1024 * 1024))
                      + " MiB locked host memory");
        } else {
            log::warn("[kimi_load] expert pinning unavailable — "
                      "keeping file-backed mmap sources");
        }
    }
```

- [ ] **Step 5: Green + negative-case тест**

Новый TEST_CASE рядом:

```cpp
TEST_CASE("kimi expert pinning: LSX_EXPERT_PIN=0 falls back to mmap",
          "[kimi][loader][pin]")
{
    auto model = BuildConvertedFixture();
    int overridden = 0;
#ifdef _WIN32
    overridden += _putenv_s("LSX_EXPERT_PIN", "0") == 0;
#else
    setenv("LSX_EXPERT_PIN", "0", 1);
    ++overridden;
#endif
    auto db = load_kimi_expert_index(model, 3, 2);
    REQUIRE(db.pinned_pool == nullptr);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    ensure_kimi_expert_resident(db, 1, 0, stream);
    REQUIRE(db.entries[1 * 2 + 0].host_w1 != nullptr);   // mmap view works
    free_kimi_expert_index(db);
    cudaStreamDestroy(stream);
#ifdef _WIN32
    _putenv_s("LSX_EXPERT_PIN", "");
#else
    unsetenv("LSX_EXPERT_PIN");
#endif
    (void)overridden;
}
```

```bash
cmake --build build/linux-release-znver4 -j 160 --target test_kimi_moe_loader 2>&1 | tail -1
LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 ./build/linux-release-znver4/tests/test_kimi_moe_loader 2>&1 | tail -3
```
Expected: все assertion зелёные, pinned store лог `pinned routed-expert store: ... experts`.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/kimi_moe_load.h libs/lsxcommon/src/kimi_moe_load.cpp tests/test_kimi_moe_loader.cpp
git commit -m "perf: kimi pinned routed-expert store (LSX_EXPERT_PIN=0 opt-out)"
```

---

### Task 3: Замер с pinned-стором + диагностика деградации

**Files:**
- Create: `/tmp/opencode/kimi_r3_*.log`, возможно `/tmp/opencode/kimi_r3.nsys-rep`

**Interfaces:**
- Consumes: Task 1 vault, Task 2 pin pass; `test_blackwell_perf_kimi` / `lsxbenchmark --model`.
- Produces: финальный t/s, вердикт «потолок/причина», список выжатых рычагов.

- [ ] **Step 1: Полный пересбор + no-env замер**

```bash
cmake --build build/linux-release-znver4 -j 160 2>&1 | grep -cE "error|FAILED" || true
LSX_TEST_KIMI_MODEL_PATH=/mnt/llm_models/models/Kimi-K2.7-Code \
  timeout 900 ./build/linux-release-znver4/tests/test_blackwell_perf_kimi 2>&1 | grep -E "PERF|pinned|expert cache" 
```
Expected: лог `pinned routed-expert store: 23040 experts, ~558000 MiB`; tok/s в 2–4× выше baseline 1.3. Записать число.

- [ ] **Step 2: Верификация корректности на реальном выводе**

```bash
./build/linux-release-znver4/apps/lsxbenchmark/lsxbenchmark \
  --model /mnt/llm_models/models/Kimi-K2.7-Code --max-gen 256 2>&1 | tail -5
```
Expected: связный текст ("Paris..."-канон), не мусор; t/s согласован с шагом 1.

- [ ] **Step 3: Анализ «деградация vs потолок»**

Потолок = эффективный pinned H2D (~45–50 GiB/s, DeepSeek-замер) ÷ miss-байты/токен (из лога кэша: misses × 24.77 MiB). Если t/s < 80% потолка:
1. `nsys profile -o /tmp/opencode/kimi_r3 --trace=cuda ./build/.../test_blackwell_perf_kimi` — искать серийность копий (copy_stream vs main stream), простои.
2. Проверить slot-бюджет: лог `expert cache: N slots` — если slot ≈ 48 MiB (2× эксперта) — найти двойную резервацию (`KimiMoeExpertDeviceCache` alloc), поднять slots.
3. Эксперименты env: `LSX_KIMI_EXPERT_CACHE_SLOTS` (0.5×/1×/2× от дефолта).
Каждая гипотеза — замер до/после; бесполезные откатываются (дисциплина DeepSeek-раундов).

- [ ] **Step 4: Зафиксировать находки в плане (этот файл), коммит промежуточный при изменениях кода**

---

### Task 4: SelfCheck ×3 + fast-сьют

- [ ] **Step 1: Реальный селфчек Paris, 3 бит-точных цикла**

```bash
LSX_TEST_KIMI_MODEL_PATH=/mnt/llm_models/models/Kimi-K2.7-Code \
  ./build/linux-release-znver4/tests/test_kimi_moe_selfcheck 2>&1 | tail -4
```
Expected: PASS ×3, `r.selfcheck_passed`, exit 0 (SKIP 4 недопустим при наличии vault).

- [ ] **Step 2: Fast-сьют целиком**

```bash
ctest --test-dir build/linux-release-znver4 -L fast 2>&1 | tail -3
```
Expected: 100% tests passed, 0 failed (skips только там, где нет vault-а других моделей).

- [ ] **Step 3: Guards**

```bash
ctest --test-dir build/linux-release-znver4 -R "test_model_independence_guard|test_code_naming_guard|test_absl_format_guard" --output-on-failure 2>&1 | tail -3
```
Expected: зелёные без правок реестров (kimi consumer зарегистрирован ранее).

---

### Task 5: Финализация (baseline JSON + AGENTS.md + docs + push)

- [ ] **Step 1: perf baseline kimi (kAdd/kUpdate)**

Добавить kimi-запись в `tests/perf/perf_baselines.json` (gemma-формат: model_type `kimi-k2.7-code`, gpu_key `rtx 5090`, os `linux`, tokens_per_sec = финальный замер, max_gen 2048, prompt канонический "The capital of France is", date, commit `-`), затем:

```bash
ctest --test-dir build/linux-release-znver4 -R test_perf_baseline --output-on-failure 2>&1 | tail -3
```
Expected: kUpdate/kAdd, тест зелёный.

- [ ] **Step 2: AGENTS.md kimi-запись** — обновить буллет `kimi-k2.7-code` (новый t/s, pinned store, формат r2 NVFP4-эксперты; `deepseek_inference_*.cu`-стиль путей выверить по факту файлов).

- [ ] **Step 3: Findings в этот план-док** (final-таблица этапов, потолок, выжатые рычаги, что осталось) + commit + push:

```bash
git add tests/perf/perf_baselines.json AGENTS.md docs/superpowers/plans/2026-09-12-kimi-nvfp4-full-contour.md
git commit -m "perf: kimi-k2.7-code pinned routed-expert store — <T> tok/s (final)"
git push
```

## Финальные результаты (2026-09-12)

| Метрика | Было (round-2) | Стало | Прирост |
|---|---|---|---|
| Decode t/s (бит-точный, дефолт) | 1.3 | **3.1** | 2.4× |
| Decode t/s (TC opt-in, tolerance) | — | 5.5 | 4.2× |
| Загрузка expert store (542 GiB) | — | 302 с | — |
| SelfCheck Paris ×3 | ✅ | ✅ | — |
| VRAM resident (dense+MLA) | 8.9 GiB | 8.9 GiB | — |
| Expert cache slots | 595 | 809 | +36% |

### Выжатые рычаги (коммиты сессии)
1. **Pinned expert store** 542 GiB в 34×16 GiB locked чанках — драйвер не
   даёт один cudaHostAlloc >~300 GiB (12-минутный reclaim-шторм, probe:
   mlock проходит, chunked cudaHostAlloc проходит). `LSX_EXPERT_PIN=0`
   отключает (927c4fe, 0d84844).
2. **TC-batch не включался** — `cfg_.fast_math` legacy bool всегда false при
   CLI; tri-state игнорировался. Фикс резолва (751aa44) дал 3.0→5.1 t/s.
3. **TC-путь сломан на реальном чекпойнте** — tolerance-class act-quant
   уводит greedy top-1 в мусор ('碧芙碧芙' вместо Paris), синтетический
   tolerance-тест при этом зелёный. Переведён в opt-in `LSX_KIMI_TC=1`
   до root-cause (14:23 фикс). Дефолт — бит-точная legacy GEMV chain.
4. **kv_b dequant thrash** — static-memo слот на ВСЕ 61 слой ре-деквантил
   kv_b 60×/токен; per-layer кэш в KimiLayerWeights (30% GPU-окна).
5. **Split-fence upload** — layout [w1|w3][s1|s3][w2][s2], ev13/ev2:
   w13 GEMM стартует, не дожидаясь down-upload (7c2058b).
6. **Резерв VRAM 8→2 GiB** (измерено: runtime-аллокации <1 GiB) →
   549→809 слотов (7c2058b).
7. **Скоростная загрузка**: 635→~280 с. Parallel parse 32 потока
   (450→8 с, c332d53); zerofill cudaHostAlloc (332 с, сериализован
   драйвером, выдавливал page cache → копия с диска) заменён на
   mmap → copy fault-at-write из горячего кэша → register последним
   (b76c76e + register-after-copy).

### Измеренная анатомия токена (nsys, TC on, 2.6 t/s снапшот)
- H2D 53% (PCIe ~50-57 GiB/s — насыщение, эксперты 200 MiB/токен cold)
- Nvfp4Dequant 30% → после kv_b-фикса ~0 (декод)
- Экспертные GEMV: cuBLASLt nvjet_**tmaAB** (TMA уже задействован)
- gaps 7% (route split-phase sync)

### Потолок
Miss-трафик ~7-11 GiB/токен при 809/23424 слотах; физический предел
PCIe RTX 5090 (~57 GiB/s) ≈ 4-6 t/s cold. 3.1 бит-точных = 65% потолка
(хвост: route sync, GEMV compute). Качественный рост дальше — только
точнее попадать в кэш (hit-rate) или wider bus.

### Root cause TC-batch divergence (2026-09-12, test_kimi_moe_tc_diag)

Дифференциальная диагностика на реальном входном векторе MoE-блока
(дамп через LSX_KIMI_DUMP_L1):
- TC-ядра корректны на квантованных данных: w13 rel 0.16-0.19%,
  down rel 0.19%, reduce rel 0.17%.
- Квантирование x (post-norm hidden): rel 12% — приемлемо.
- **Квантирование swiglu-выхода (hq): rel 120%** — E2M1 per-16 pow2
  превращает тензор в шум: у swiglu-выхода тяжёлый хвост + масса
  значений у нуля, блоки неравномерны.
- Вклад hq-кванта в down-выход: **69%** — совпадает с полным
  расхождением legacy vs TC (67%).
- Вывод: формат mxf4nvf4 (per-16 E2M1 активации) фундаментально
  непригоден для swiglu-тензоров Kimi. TC-batch остаётся opt-in;
  дефолт — бит-точная legacy GEMV (3.1 t/s). Путь к 5.5 t/s — формат
  активаций повышенной точности для down-GEMM (HW HW-формат per-16
  ue4m3 уже полный E4M3 — но E2M1-нибблы данных остаются узким
  местом; нужен hw-путь с BF16 A-операндом, которого в mxf4nvf4 нет).

### Бэклог
- Пиндить expert store по умолчанию для ВСЕХ моделей (768 ГБ ОЗУ)
- Unify DeepSeek/Kimi pin-pass в ExpertStore::PinAll
- TC-путь Kimi: поднять точность акт-кванта down-GEMM (см. root cause)
