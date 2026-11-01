# Аудит работы с Arrow во всех адаптированных моделях — дизайн

Дата: 2026-11-01
Ветка: `audit/arrow-usage-2026-11-01`
Статус: дизайн утверждён оператором; реализация — по TDD.

## 1. Задача

Проверить, как каждая адаптированная модель читает и пишет Apache Arrow IPC,
найти дефекты корректности/безопасности/производительности и закрыть их
регрессионными тестами. Легаси-паттерн «pinned arena со всеми экспертами»
(`UnpackBatch` → `std::vector` → `cudaMallocHost`) **не мигрируется** в этой
ветке: он уже задокументирован как долг в `docs/OPEN_TASKS.md`; здесь он
описывается в отчёте и удерживается стражем.

Исходные данные: 29 исходных чекпойнтов в `/mnt/fast_storage/huggingface`,
у 26 есть сконвертированное дерево `logestix/` (нет у DeepSeek-V4-Flash,
DeepSeek-V4.1-Flash, Qwen3.5-35B-A3B).

## 2. Инвентарь (что именно аудируется)

Центральный слой (один на все семейства):

| файл | роль |
|---|---|
| `libs/lsxcommon/src/schema.cpp` | `Expert/SharedExpert/DenseNvfp4/Pinned/Tokenizer*Schema` (`MakeBatch`/`UnpackBatch`), `WriteRecordBatchToFile`, `ReadRecordBatchFromFile`, `Load*BatchViews` |
| `libs/lsxcommon/include/lsxcommon/expert_store.h` | нейтральный host-стор: mmap + `RecordBatchFileReader` + LRU, `DecodeEntry`, `PinChunked` |
| `libs/lsxcommon/src/shared_experts.cu` | прямой zero-copy mmap shared-experts |
| `libs/lsxcommon/src/tokenizer.cpp` | эталон R2: `in_range`-проверки каждого буфера против mmap |
| `libs/lsxcommon/src/model_integrity.cpp` | проверка построенных токенизаторных артефактов |

24 загрузчика используют `LoadTensorBatchViews` / `LoadDenseNvfp4BatchViews`,
3 — `LoadSharedExpertBatchViews`; 6 легаси-арен идут через
`ReadRecordBatchFromFile` + `UnpackBatch`; 24 конвертера пишут через
`WriteRecordBatchToFile` (или собственные writer-циклы).

Типы файлов в деревьях: `experts/layer_N/exp_E.arrow`,
`pinned/{global,layer_N,layer_N_nvfp4,global_nvfp4,shared_experts,
layer_N_adapters}.arrow`, `shared/layer_N_{nvfp4,gate}.arrow`,
`router/layer_N.arrow`, `mtp/{raw,nvfp4}.arrow`, `indexer/layer_N.arrow`,
`ple/{layer_N,layer_N_nvfp4,meta,shard_N}.arrow`,
`pinned/tokenizer_{vocab,merges}.arrow`.

Правила, которые аудит должен подтвердить или ввести:

- **R-A1** Чтение — только `MemoryMappedFile::Open`; `ReadableFile::Open`
  запрещён (нет zero-copy и нет mmap-keepalive).
- **R-A2** Retained payload — только view-API (`Load*BatchViews`,
  `ExpertStore`, `shared_experts`, `tokenizer`) либо `UnpackBatch` в
  transient-скоупе. `UnpackBatch`-call-site перечислен в страже с вердиктом.
- **R-A3** Любой опубликованный указатель обязан лежать в границах своего
  Arrow-буфера (data, scale list values, shape list values).
- **R-A4** Схема валидируется по именам/типам, а не только по числу колонок.
- **R-A5** Бинарный payload обязан быть согласован с геометрией: `bytes ==
  product(shape) * dtype_size`; NVFP4: `packed == ceil(numel/2)`,
  `scales == ceil(numel/16)`, `global_scale` конечен и > 0.
- **R-A6** Запись артефакта атомарна: сбой не оставляет валидный с виду файл.
- **R-A7** `RecordBatch` из `MemoryMappedFile` держит маппинг живым через
  цепочку `Buffer → parent → Region` (проверено по исходникам Arrow 26);
  явный keepalive view-API всё равно обязателен как контракт.
- **R-A8** Каждое семейство имеет детерминированную строку в матрице аудита
  (конвертер → загрузчик → путь чтения → вердикт).

## 3. Находки (предварительные, подтверждаются тестами)

| ID | P | Где | Суть |
|---|---|---|---|
| A-01 | P1 | `schema.cpp` (`ExpertSchema`, `SharedExpertSchema`, `DenseNvfp4Schema`, `PinnedSchema`), `ExpertStore::DecodeEntry` | owning-пути не проверяют `num_rows`, типы list-values/offsets и границы offsets → OOB-чтение или null-deref на битом/чужом `.arrow`. X15 закрыл только view-путь |
| A-02 | P1 | `LoadTensorBatchViews`, `LoadDenseNvfp4BatchViews` | нет сверки payload с `shape`/`dtype` и NVFP4-инвариантов → усечённый файл проходит, H2D читает за границей mmap (24 загрузчика) |
| A-03 | P2 | `ShapePtr` (view-путь) | offsets shape-списка не проверяются против длины values (X15 покрыл data/scales) |
| A-04 | P2 | `shared_experts.cu` | `end < begin` даёт отрицательный размер; нет проверки монотонности offsets |
| A-05 | P2 | `WriteRecordBatchToFile` | не атомарна: сбой посреди записи оставляет частичный `.arrow`, который затем читается как валидный |
| A-06 | P3 | `ReadRecordBatchFromFile` | нет проверки схемы/числа строк; ошибка всплывает поздно как «unpack failed» |
| A-07 | P3 | `gemma_moe_load.cpp`, `gemma_dense_load.cpp` | собственный `LoadArrowFile` копирует через `UnpackBatch` вместо view-хелперов (перф, не корректность) |
| A-08 | P3 | 6 легаси-арен | документированный долг; отчёт + страж, миграции нет |
| A-09 | ok | `tokenizer.cpp`, `ExpertStore::ParseFile` | эталон: `in_range` по каждому буферу, mmap+batch удерживаются |

## 4. Реализация

### 4.1 Общий слой проверок

Новый host-only заголовок `libs/lsxcommon/include/lsxcommon/arrow_validate.h`
(inline, без abseil-контейнеров, безопасен для включения из `.cu`):

- `CheckListSlice(const arrow::Int32Array&, row, values_length, what, name)` —
  монотонность и границы;
- `CheckViewRange(ptr, bytes, buffer, what, name)` — перенос существующей
  проверки X15 из `schema.cpp`;
- `DtypeByteSize(std::string_view dtype)` — BF16/F16/F32/F64/I8/U8/I32/U32/
  I64/U64/F8_E4M3/E2M1; неизвестный dtype → ошибка;
- `CheckTensorSize(shape, rank, bytes, dtype, name)` — R-A5;
- `CheckNvfp4Size(shape, packed_bytes, scale_bytes, gs, name)` — R-A5 для
  NVFP4.

`schema.cpp` использует их в `UnpackBatch`-путях и `ShapePtr`;
`expert_store.h::DecodeEntry` — для `num_rows == 1`, типов и границ;
`shared_experts.cu` — для scale-offsets.

### 4.2 Ужесточение owning-путей и view-загрузчиков

- `ExpertSchema::UnpackBatch`: `num_rows == 1`, проверка типов каждой
  колонки, `CheckListSlice` для scale, `CheckViewRange` для бинарей.
- `SharedExpertSchema::UnpackBatch`: проверка `values()`/`offsets()` на null
  (сейчас падает), `CheckListSlice` на каждую строку.
- `DenseNvfp4Schema::UnpackBatch`: `CheckListSlice` для shape и scales,
  `CheckTensorSize`/`CheckNvfp4Size`.
- `PinnedSchema::UnpackBatch`: `CheckListSlice` для shape, `CheckTensorSize`.
- `ExpertStore::DecodeEntry`: `num_rows == 1`, null-safe, `CheckListSlice`.
- `LoadTensorBatchViews`/`LoadDenseNvfp4BatchViews`: `CheckTensorSize` /
  `CheckNvfp4Size` на каждую запись, `ShapePtr` с проверкой границ.
- `shared_experts.cu`: полная проверка offsets.

### 4.3 Атомарная запись

`WriteRecordBatchToFile(batch, path)`:

1. `path.parent_path()` должен существовать (иначе ошибка, без создания);
2. писать в `path + ".tmp.<pid>.<counter>"`;
3. `writer->Close()` и `outfile->Close()` проверяются;
4. `fs::rename(tmp, path)` (atomic replace на POSIX; на Windows —
   `MoveFileEx`-семантика `fs::rename` с предварительным remove, только для
   тестового пути);
5. на любой ошибке — удалить tmp и вернуть статус.

### 4.4 Статический страж

`tests/test_arrow_usage_guard.cpp` (label `fast`, host-only, как
`test_expert_store_guard`):

1. В `libs/` и `apps/` нет `arrow::io::ReadableFile::Open`.
2. Каждый `MemoryMappedFile::Open` встречается только в разрешённых файлах
   (schema.cpp, expert_store.h, tokenizer*.cpp, model_integrity.cpp,
   shared_experts.cu, gemma_*_load.cpp — до фикса A-07).
3. Каждый вызов `*Schema::UnpackBatch` перечислен в таблице
   `{файл, строка-владелец, вердикт transient|retained-legacy}`; новый
   call-site без строки валит тест.
4. Статический реестр 29 моделей: `model dir → converter TU → loader TU →
   report row`; каждая модель обязана присутствовать в
   `docs/ARROW_AUDIT_2026-11-01.md` (проверка по маркеру строки).
5. `WriteRecordBatchToFile` не вызывается напрямую из конвертеров в обход
   schema.cpp (разрешены только schema.cpp и тестовые фикстуры).

### 4.5 Runtime-валидатор деревьев

`tests/test_arrow_tree_integrity.cpp` (label `convert`, `SKIP_RETURN_CODE 4`,
`TIMEOUT 1800`):

- корень: `LSX_ARROW_AUDIT_ROOT`, по умолчанию
  `/mnt/fast_storage/huggingface`; отсутствие корня или `logestix/` → `SKIP`;
- для каждого дерева: `model.index.json` (валидный JSON, известный `model`),
  обход всех `.arrow`;
- дешёвая фаза: `MemoryMappedFile::Open` + `RecordBatchFileReader::Open`
  (только footer/схема/block-table), `num_record_batches() == 1`,
  классификация схемы по именам полей, `num_rows > 0`;
- глубокая фаза (детерминированная выборка: первый и последний файл каждого
  каталога/слоя, плюс `LSX_ARROW_AUDIT_SAMPLE` файлов на дерево): полный
  `ReadRecordBatch(0)` + все проверки R-A3/R-A5;
- `LSX_ARROW_AUDIT_FULL=1` — глубокая фаза для всех файлов (ручной прогон);
- бюджет: не более `LSX_ARROW_AUDIT_MAX_BYTES` (по умолчанию 2 ГиБ)
  прочитанного payload на дерево; превышение — `SKIP` фазы с записью в лог.

### 4.6 Отчёт

`docs/ARROW_AUDIT_2026-11-01.md` (русский, формат
`docs/INFERENCE_AUDIT_2026-10-22.md`):

- что проверено, инвентарь по 29 моделям;
- матрица «модель → конвертер → загрузчик → путь чтения (mmap/view/unpack) →
  путь записи → вердикт»;
- находки A-01…A-09 со ссылками `file:line`, доказательствами и статусом;
- таблица легаси-долга (host RAM, что не мигрировано и почему);
- результаты прогонов (fast, новые тесты, валидатор 26 деревьев);
- команды проверки.

## 5. TDD-порядок

1. `tests/test_arrow_validation.cpp` (label `fast`) — RED: рукотворные битые
   `RecordBatch` (0 строк, >1 строки, отрицательные/завышенные offsets,
   list с не-uint8 values, shape×dtype ≠ bytes, усечённый NVFP4,
   `global_scale` NaN) против всех `UnpackBatch` и `DecodeEntry`.
2. `arrow_validate.h` + фиксы — GREEN по одному кейсу.
3. `tests/test_arrow_validation.cpp` — RED для view-хелперов (A-02/A-03),
   затем фикс.
4. `tests/test_schema_views.cpp` — добавить кейс атомарной записи (A-05),
   затем фикс `WriteRecordBatchToFile`.
5. `tests/test_arrow_usage_guard.cpp` — RED на текущем коде (A-07 + новые
   call-sites), затем фикс и allowlist.
6. `tests/test_arrow_tree_integrity.cpp` — прогнать на 26 деревьях; любая
   ошибка = находка, фиксируется в отчёте (фикс — если это код, а не
   артефакт; испорченный артефакт не перезаписывается).
7. Отчёт + `AGENTS.md` (строка о новом страже), финальный прогон.

## 6. Проверка

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug -R "test_arrow_|test_schema_views|test_expert_store"
ctest --preset linux-debug -L fast
ctest --preset linux-debug -R test_arrow_tree_integrity   # 26 деревьев
```

Временные файлы — только в `/mnt/fast_storage/tmp`; сборка — существующие
пресеты, без изменения флагов.

## 7. Вне рамок

- миграция 6 легаси-арен на `ExpertStore` (отдельная работа);
- переписывание safetensors-парсера конвертеров (не Arrow);
- изменение формата/схем на диске (формат стабилен; аудит его только
  валидирует);
- производительность view-пути (только A-07 как дешёвый фикс).

## 8. Риски

- Валидатор на 800-гигабайтных деревьях может уйти в I/O: поэтому footer-only
  по умолчанию и бюджет payload.
- Ужесточение схем может отвергнуть существующее дерево, которое раньше
  «работало»: любое такое расхождение разбирается как отдельная находка, а не
  ослаблением проверки.
- Атомарная запись на Windows: `fs::rename` не заменяет существующий файл;
  используем remove-then-rename только под `_WIN32`, тесты гоняются на Linux.
