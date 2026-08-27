# Спека: Kimi MoE — device-кэш экспертов + батч-пайплайн (единая VRAM-политика)

Дата: 2026-08-27
Статус: утверждена оператором
Предпосылка: централизованная политика резидентности экспертов
(`lifecycle::ExpertCacheSlots`, `kExpertCacheVramReserveBytes = 4 GiB` в
`resource_governor.h`) подключена к qwen-moe, gemma, deepseek. Kimi стримит
экспертов per-call без device-кэша — эта спека закрывает последний пробел.

## 1. Проблема

Kimi (61 слой × 384 эксперта × top-8, эксперт ≈ 22 МБ INT4-упаковка, всего
≈ 515 ГБ) грузит веса экспертов с host-mmap (`ExpertStore`) на устройство
на каждый токен: 488 экспертов × H2D-копия 22 МБ + 3 GEMV, синхронно по
одному (`RunKimiExpertFfn`, цикл в `kimi_moe_inference.cu:947-959`).
≈ 10 ГБ PCIe-трафика на токен — доминирующее узкое место decode.

## 2. Решение

### 2.1 Device slot-кэш (`kimi_moe_expert_cache`)

- `include/lsxcommon/kimi_moe_expert_cache.h` — host-bookkeeping **без CUDA**:
  - слоты из `lifecycle::ExpertCacheSlots(free, per_expert, total)`
    (per_expert = w1+w3+w2+scales ≈ 22 МБ, total = 61×384 = 23424);
  - явный override `LSX_KIMI_EXPERT_CACHE_SLOTS` (паритет с другими
    семействами); значение 0 = кэш отключён, legacy per-call staging путь;
  - LRU-выселение, плоский ключ `layer * 384 + expert`;
  - decision-API `ResolveBatch(layer, top8_ids)` → `{resident[slot], missing[]}`
    — резидентность всего top-8 слоя за один вызов; повторный вызов для тех же
    экспертов идемпотентен (все resident);
  - decision-API не аллоцирует (zero-alloc контракт).
- `src/kimi_moe_expert_cache.cu` — CUDA-часть: аллокация арены слотов
  (pack + scale strides), батчевый async-upload missing на copy-stream,
  event-фенс перед чтением слота, LRU-эвикция с синхронизацией copy-stream
  (in-flight DMA должен завершиться до перезаписи слота).
- Слот начинается с prewarm-нули: до первого заполнения слот невалиден.

### 2.2 Батч-пайплайн в генерации

Цикл top-8 слоя (`kimi_moe_inference.cu:947-959`) заменяется на:

1. Pass 1 — `ResolveBatch(layer, top8_ids)` для всех 8 экспертов;
2. Pass 2 — батчевый async-upload всех missing на copy-stream одним пакетом
   (w1|w3|w2 + scales в слоты кэша); параллельно на compute-stream
   выполняются GEMV уже-резидентных экспертов;
3. Pass 3 — event-wait завершения upload, GEMV оставшихся из слотов.

Порядок GEMV-аккумуляции в `acc` сохраняется (детерминизм; очередность
обработки top-k не влияет на результат — сложение в FP32-аккумулятор
выполняется в фикс. порядке индексов k, не в порядке завершения копий).

### 2.3 Fallback и деградация

- Аллокация кэша при старте: shrink-retry вдвое (паритет qwen/gemma);
  0 слотов → legacy per-call staging путь (текущее поведение) + warn.
- Явный `LSX_KIMI_EXPERT_CACHE_SLOTS=0` — тот же legacy-путь, без warn.

## 3. Бит-точность

GEMV-ядро читает идентичные упакованные байты из слота кэша вместо
staging-буфера — вход бит-в-бит совпадает, ядро то же ⇒ выходы бит-точны.
Копия на устройство не трансформирует данные. Самое лучшее доказательство —
реальный selfcheck `test_kimi_moe_selfcheck` (на машинах с моделью): bit-exact
детерминизм `final_logits` сохраняется (тест SKIP без модели).

## 4. Тесты (TDD)

- Новый `tests/test_kimi_moe_expert_cache.cpp` (host-only, Catch2):
  - policy-wiring: слоты из `lifecycle::ExpertCacheSlots`, clamp до 23424;
  - env override: explicit > policy; `0` = disable;
  - LRU: порядок выселения при переполнении, переиспользование горячего;
  - `ResolveBatch`: классификация resident/missing, идемпотентность,
    обработка невалидных индексов (отрицательные, вне диапазона → invalid);
  - zero-alloc контракт decision-API;
  - план батч-upload: байтовые диапазоны missing покрывают все записи
    эксперта (w1, w3, w2, scales) и не пересекаются.
- Регресс: существующие host-тесты `test_kimi_*` зелёные; naming guard
  зелёный (новые имена по схеме, allowlist не расширяется).

## 5. Сборка

- `src/kimi_moe_expert_cache.cu` → OBJECT `lsxcommon_kimi_cu`
  (`libs/lsxcommon/CMakeLists.txt`), флаги объекта не меняются
  (`--extended-lambda; -fmad=false`).
- Новый тест в `tests/CMakeLists.txt` (host-only, без lsxcommon CUDA-целей —
  только bookkeeping-заголовок).
- Полная пересборка windows-debug exit=0.

## 6. Критерии готовности

- `test_kimi_moe_expert_cache` зелёный; policy-тесты governor зелёные;
  naming guard зелёный; полный host-набор без регрессий.
- Лог-строка паритета при старте: `[kimi-infer] expert cache: free=...
  per_expert=... total=... slots=...`.
- На машинах с моделью Kimi: selfcheck бит-точен 3 цикла (вне охвата этой
  машины, отмечено как follow-up верификация).
- Явное ограничение зафиксировано в логах/комментарии: при 16 ГиБ VRAM кэш
  покрывает ~2% экспертов — эффект зависит от горячести роутинга; main win
  в любом случае — батчевый upload вместо 8 синхронных H2D на слой.
