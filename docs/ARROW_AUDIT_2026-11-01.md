# Аудит работы с Apache Arrow — 2026-11-01

Ветка `audit/arrow-usage-2026-11-01` (от `main` @ `465f6063`). Хост: Linux,
CUDA 13.4, сборка `linux-debug` (`cmake --preset linux-debug`).
Исходные чекпойнты: `/mnt/fast_storage/huggingface` (29 моделей, у 26 есть
сконвертированное дерево `logestix/`).

Внешние оракулы:

- реальные сконвертированные деревья (26 моделей, **273 254** файла `.arrow`);
- исходники Apache Arrow 26.0 из `build/linux-debug/_deps/arrow-src`
  (семантика `MemoryMappedFile::ReadAt` → `Region`/`Slice`: zero-copy буфер
  держит маппинг живым через `Buffer::parent`, даже если поток закрыт);
- регрессионные тесты `tests/test_arrow_validation.cpp`,
  `tests/test_arrow_usage_guard.cpp`, `tests/test_arrow_tree_integrity.cpp`.

## 1. Что проверено

| Слой | Файлы |
|---|---|
| схемы и I/O | `libs/lsxcommon/src/schema.cpp` (`Expert/SharedExpert/DenseNvfp4/Pinned/Tokenizer*Schema`, `WriteRecordBatchToFile`, `ReadRecordBatchFromFile`, `Load*BatchViews`) |
| host-стор экспертов | `libs/lsxcommon/include/lsxcommon/expert_store.h` (`ParseFile`, `DecodeEntry`, `PinChunked`) |
| прямой mmap | `libs/lsxcommon/src/shared_experts.cu` |
| эталон R2 | `libs/lsxcommon/src/tokenizer.cpp` (`in_range` по каждому буферу) |
| 24 загрузчика | `LoadTensorBatchViews` / `LoadDenseNvfp4BatchViews` |
| 3 загрузчика | `LoadSharedExpertBatchViews` |
| 6 легаси-арен | `ReadRecordBatchFromFile` + `UnpackBatch` |
| 24 конвертера | запись через `WriteRecordBatchToFile` |

Типы файлов в деревьях: `experts/layer_N/exp_E.arrow`,
`pinned/{global,layer_N,layer_N_nvfp4,global_nvfp4,shared_experts,
layer_N_adapters}.arrow`, `shared/layer_N_{nvfp4,gate}.arrow`,
`router/layer_N.arrow`, `mtp/{raw,nvfp4}.arrow`, `indexer/layer_N.arrow`,
`ple/{layer_N,layer_N_nvfp4,meta,shard_N}.arrow`,
`pinned/tokenizer_{vocab,merges}.arrow`. Все 7 схем классифицируются по именам
полей; неизвестная схема — ошибка валидатора.

## 2. Матрица «модель → код → путь чтения»

`verdict` — тип host-хранения routed-экспертов (dense-модели помечены
`views+dense`). Плотные (pinned) тензоры во всех семействах читаются
zero-copy view-хелперами; исключение — gemma (см. A-07).

<!-- arrow-matrix:start -->
| model | converter | loader | read path | write path | verdict |
|---|---|---|---|---|---|
| AliceAI-Foundation-80B-A3B-Base | alice_moe_convert.cpp | alice_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Beam-1.0-501B-A23B | beam_moe_convert.cpp | beam_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| DeepSeek-V4-Flash | deepseek_moe_repack.cpp | deepseek_moe_load.cu | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| DeepSeek-V4-Pro-0813 | deepseek_moe_pro_convert.cpp | deepseek_moe_pro_load.cu | views + neutral ExpertStore (mmap-stream) | schema.cpp | views+ExpertStore |
| DeepSeek-V4.1-Flash | deepseek_moe_v41_convert.cpp | deepseek_moe_v41_load.cu | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| GLM-5.3-Flash | glm_moe_flash_convert.cpp | glm_moe_flash_load.cpp | unpack + pinned arena | schema.cpp | unpack+pinned-arena |
| GigaChat3.5-432B-A28B-Reasoning | gigachat_moe_convert.cpp | gigachat_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Hy-Memory-30B-A3B | hy_moe_convert.cpp | hy_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| Hy-SimulMT-30B-A3B | hy_moe_simulmt_convert.cpp | hy_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| Kimi-K2.7-Code | kimi_moe_convert.cpp | kimi_moe_load.cpp | views + PinChunked | schema.cpp | views+PinChunked |
| Ling-3.1-flash | ling_moe_convert.cpp | ling_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| MiniMax-M3 | minimax_moe_convert.cpp | minimax_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| MiniMax-M3.1-Flash | minimax_moe_flash_convert.cpp | minimax_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Mistral-Small-4-119B-2603-NVFP4 | mistral_moe_convert.cpp | mistral_moe_load.cpp | unpack + pinned arena (MoEExpertStore) | schema.cpp | unpack+pinned-arena |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | nemotron_moe_convert.cpp, nemotron_moe_nvfp4_convert.cpp | nemotron_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Ornith-1.5-35B-A3B | ornith_moe_convert.cpp | ornith_moe_load.cpp | unpack + pinned arena (MoEExpertStore) | schema.cpp | unpack+pinned-arena |
| Qwen3.5-35B-A3B | qwen_moe_convert.cpp | qwen_moe_load.cpp | unpack + pinned arena (MoEExpertStore) | schema.cpp | unpack+pinned-arena |
| Qwen3.5-9B | qwen_dense_convert.cpp | qwen_inference_load.cpp | views + dense | schema.cpp | views+dense |
| Qwen3.6-35B-A3B | qwen_moe_36_convert.cpp | qwen_moe_load.cpp | unpack + pinned arena (MoEExpertStore) | schema.cpp | unpack+pinned-arena |
| Qwen3.8-27B | qwen_dense_convert.cpp | qwen_inference_load.cpp | views + dense | schema.cpp | views+dense |
| Qwen3.8-Flash-Next | qwen_moe_exp_convert.cpp | qwen_moe_exp_load.cpp | unpack + pinned arena | schema.cpp | unpack+pinned-arena |
| Step-5-Preview | step_moe_convert.cpp | step_moe_load.cpp | views + neutral ExpertStore | schema.cpp | views+ExpertStore |
| Xing4.0-29B-A4B | xing_moe_convert.cpp | xing_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| flint | flint_moe_convert.cpp | flint_moe_load.cpp | views + resident device slabs | schema.cpp | views+resident-slabs |
| gemma-4-26B-A4B-it | gemma_moe_convert.cpp | gemma_moe_load.cpp | views + PinChunked (dense triple: views) | schema.cpp | views+PinChunked |
| gemma-4-31B-it | gemma_dense_convert.cpp | gemma_dense_load.cpp | views + dense (triple: views) | schema.cpp | views+dense |
| granite-4.2-30b | granite_dense_convert.cpp | granite_dense_load.cpp | views + dense | schema.cpp | views+dense |
| granite-switch-4.1-30b | granite_dense_switch_convert.cpp | granite_dense_switch_load.cpp | views + dense (+LoRA planes) | schema.cpp | views+dense |
| phi-4 | phi_dense_convert.cpp | phi_dense_load.cpp | views + dense | schema.cpp | views+dense |
<!-- arrow-matrix:end -->

DeepSeek-V4-Flash, DeepSeek-V4.1-Flash и Qwen3.5-35B-A3B дерева в
`/mnt/fast_storage/huggingface` не имеют — для них проверен только код
(конвертер/загрузчик/схемы), не артефакты.

## 3. Находки

| ID | P | Где | Суть | Статус |
|---|---|---|---|---|
| A-01 | P1 | `schema.cpp:148,339,470,592`, `expert_store.h:195` | owning-пути не проверяли `num_rows`, типы list-values/offsets и границы offsets → OOB-чтение; `SharedExpertSchema::UnpackBatch` падал SIGSEGV на list с не-`uint8` | fixed |
| A-02 | P1 | `schema.cpp:1025,1076` | view-загрузчики не сверяли payload с `shape`/NVFP4-инвариантом → усечённый файл проходил, H2D читал за границей mmap (24 загрузчика) | fixed |
| A-03 | P2 | `schema.cpp:970` (`ShapePtr`) | offsets shape-списка не проверялись против длины values (X15 покрыл только data/scales) | fixed |
| A-04 | P2 | `shared_experts.cu:307` | reversed scale offsets давали отрицательный `scale_bytes` | fixed |
| A-05 | P2 | `schema.cpp:858` | запись не атомарна: сбой оставлял частичный `.arrow` на финальном пути | fixed |
| A-06 | P3 | `schema.cpp:916` | `ReadRecordBatchFromFile` не сверяет схему/число строк — ошибка всплывала поздно как «unpack failed» | accepted (диагностика) |
| A-07 | P3 | `gemma_moe_load.cpp`, `gemma_dense_load.cpp` | плотные тензоры читаются копией через `UnpackBatch` вместо view-хелперов (perf, не корректность) | fixed (views) |
| A-08 | P3 | 6 легаси-арен | pinned arena со всеми экспертами (host RAM = дерево экспертов) | accepted (долг) |
| A-09 | ok | `tokenizer.cpp:55`, `expert_store.h:ParseFile` | эталон: `in_range` по каждому буферу, mmap+batch удерживаются | reference |

### A-01 — owning-пути не валидировали batch

`ExpertSchema::UnpackBatch`/`SharedExpertSchema::UnpackBatch`/
`DenseNvfp4Schema::UnpackBatch`/`PinnedSchema::UnpackBatch` и
`ExpertStore::DecodeEntry` брали `offsets->Value(...)`/`GetView(0)` без проверок.
Битый или чужой `.arrow` давал чтение за буфером; `SharedExpertSchema` на
`list<int16>` разыменовывал null. Теперь все они вызывают общий
`lsxcommon/arrow_check` (`arrow_validate.h`): `CheckListSlice`, `CheckViewRange`,
`CheckTensorSize`, `CheckNvfp4Size`, `CheckPinnedTensor`; `DecodeEntry` возвращает
`arrow::Status`, а `ParseFile` отбрасывает запись. Регрессия:
`test_arrow_validation.cpp` (23 кейса, включая бывший SIGSEGV).

### A-02 — view-загрузчики не сверяли размер с геометрией

`LoadTensorBatchViews` теперь проверяет `bytes == product(shape)*dtype_size`
(таблица dtype: BF16/F16/I16/U16/F32/I32/U32/F64/I64/U64/I8/U8/F8_E4M3/F8_E5M2/
BOOL) и overflow формы; `LoadDenseNvfp4BatchViews` — NVFP4-инвариант
`packed == ceil(numel/2)`, `scales == ceil(numel/16)`, конечный
`global_scale > 0`. Это закрывает 24 загрузчика одним изменением.

### A-03 — shape-список во view-пути

`ShapePtr` проверяет пару offsets и её границы; при ошибке загрузчик
возвращает `Invalid`, а не публикует указатель.

### A-04 — `shared_experts.cu`

`extract_scales` валидирует `begin/end` через `CheckListSlice` до вычисления
указателя; отрицательный размер больше невозможен.

### A-05 — атомарная запись

`WriteRecordBatchToFile` пишет `<path>.tmp.<stamp>.<seq>`, закрывает writer и
поток с проверкой статусов, затем `rename` на финальный путь; на любой ошибке
временный файл удаляется `TempFileGuard`. `LSX_TEST_ARROW_WRITE_FAIL=1` —
тестовый шов для регрессии. На Windows — remove-then-rename (rename не
заменяет существующий файл).

## 3a. Две семантики, найденные валидатором деревьев

Это главный результат внешнего оракула: обе ошибки были бы регрессией
загрузки, если бы проверки включили «по документации», а не по реальным
деревьям.

1. **`DenseNvfp4Record.dtype` — это dtype ИСТОЧНИКА, не payload.** В деревьях
   Alice/MiMo/Minimax запись `*_nvfp4.arrow` имеет `dtype = "BF16"`/
   `"F8_E4M3"`, а `packed` — E2M1 нибблы (`numel/2`) и `scales` — E4M3
   (`numel/16`). Проверка «E2M1 → NVFP4, иначе shape×dtype» отвергла бы
   реальное дерево. Правило: payload DenseNvfp4 всегда NVFP4.
2. **Pinned NVFP4-триплет gemma.** Dense-вес gemma хранится тремя
   Pinned-записями с ЛОГИЧЕСКОЙ формой: `<name>.packed` (U8, `numel/2`),
   `<name>.scales` (U8, `numel/16`), `<name>.global_scale` (F32, `{1}`).
   Общая проверка `bytes == numel*dtype_size` отвергла бы
   `gemma-4-26B/layer_0.arrow`. Правило: суффиксы `.packed`/`.scales`
   валидируются NVFP4-инвариантом (`CheckPinnedTensor`).

3. **Блочный размер scales у routed-экспертов зависит от семейства.**
   Свежий NVFP4-квантователь даёт E4M3/16, а DeepSeek passthrough — E8M0/32
   или /128 (диагностика загрузчика печатает `blk=128 elems/scale`). Поэтому
   `ExpertSchema::UnpackBatch` проверяет только кодировко-независимые
   инварианты: целостность offsets, «оба пусты или оба непусты» (пустая запись
   = неактивный слот GLM), конечный положительный `global_scale`. Проверка
   per-16 возможна только там, где формат однозначен: `DenseNvfp4` и gemma
   `.packed`/`.scales`.

Все три семантики закреплены тестами (`test_arrow_validation.cpp`:
«DenseNvfp4 payloads are NVFP4 regardless of the source dtype», «Pinned NVFP4
triple records validate against the logical shape»; пустой слот —
существующий `test_deepseek_moe_loader` «Load expert with empty w1 fields»).

## 3b. Оптимизации пути записи (вторая итерация)

Arrow 26.0.0 в нашем узком пути (IPC + mmap + zero-copy) новых возможностей не
добавляет; выигрыш — в отказе от собственных лишних копий и в использовании
существующих API. Всё закреплено детерминированным бюджетом аллокаций
(`test_arrow_validation`, тег `[budget]`, `TrackedPool` считает пик и число
растущих реаллокаций; тайминги не используются).

| Что | До | После |
|---|---|---|
| `ExpertSchema::MakeBatch`: промежуточный `AllocateBuffer` + `memcpy`, затем builder | пик пула **2×payload** (41.96 МБ на 20 МиБ payload) | **1×payload** (20.98 МБ) |
| `PinnedSchema::MakeBatch` (32×1 МиБ) | 10 растущих реаллокаций | **0** |
| `DenseNvfp4Schema::MakeBatch` (32 записи) | 33 | **0** |
| `SharedExpertSchema::MakeBatch` (16×3 МиБ) | 39 | **0** |

- значения-буферы всех четырёх схем теперь пре-резервируются (`ReserveData`),
  а списки — `Reserve(n+1)` (финальный offset);
- `make_scales_u8`/`ExpertSchema` принимают необязательный `arrow::MemoryPool*`
  (default — пул Arrow; это шов для бюджета, поведение по умолчанию не меняется);
- `KickWholeFileReadahead` переведён с Linux-only `posix_fadvise` на
  `MemoryMappedFile::WillNeed` (портируемо, тот же madvise);
- footer-фаза `test_arrow_tree_integrity` использует
  `RecordBatchFileReader::CountRows()` (только метаданные батча, без payload):
  каждый из 273 254 файлов проверяется на непустоту, expert-файл — ровно 1
  строку. Цена — 33 с → **211 с** (лимит 600 с).

Оценка эффекта на реальной конвертации: для 811 ГБ дерева DeepSeek-V4-Pro
устраняется ~0.8 ТБ host-memcpy; для много-записных layer-файлов пик RSS записи
падает с ~2S до ~S. Формат на диске и путь чтения не меняются.

## 3c. Точная проверка экспертов и zero-copy запись (третья итерация)

- **Единый блочный контракт измерен, а не предположен**: все 19 деревьев с
  глубокой выборкой экспертов дают `block=16` (включая DeepSeek-V4-Pro).
  Поэтому `CheckExpertScaleCount` (один E4M3 на 16 элементов) применяется:
  в `ExpertStore::DecodeEntry` (по умолчанию 16), в `SharedExpertSchema` и
  `LoadSharedExpertBatchViews` (16), в `shared_experts.cu` (16) и в
  `ExpertSchema::UnpackBatch(batch, scale_block)` — параметр опционален:
  конвертеры/`deepseek_moe_repack` читают pre-repack источники (block 32) без
  проверки, все загрузчики converted-деревьев передают 16.
  Реальные деревья и 13 loader-тестов зелёные; усечённый, но валидный по
  offsets эксперт теперь отвергается.
- **`ReadRecordBatchFromFile(path, expected_columns=0)`**: нулевое число строк
  и неверная ширина отвергаются сразу, а не как позднее «unpack failed».
- **`ExpertSchema::MakeBatchView`**: zero-copy вариант батча — весовые колонки
  оборачивают векторы `PackedExpert` (`Buffer::Wrap`), копируются только
  scales/offsets/gs. Используется в `SaveExpertToArrow`,
  Mistral/MiniMax/Qwen3.8-Flash-Next конвертерах; контракт времени жизни
  задокументирован (батч живёт, пока жив источник). Бюджет: при 20 МиБ
  payload пик пула ≤ 4 МиБ (было 20 МиБ), т.е. writer — единственная копия.
- **Убрана мёртвая копия в `make_scales_u8`** (строитель `ub` нигде не
  использовался).
- **Страж `ensure_alignment`**: `test_arrow_usage_guard` запрещает выставлять
  `IpcReadOptions::ensure_alignment`/`Alignment::k{8,32,64}Byte` — это
  заставило бы Arrow копировать буферы и уничтожило zero-copy.

## 3d. Post-conversion hook и gemma zero-copy (четвёртая итерация)

- Верификатор вынесен в `lsxcommon::arrow_check::VerifyArrowTree` /
  `FindConvertedTrees` (`libs/lsxcommon/{include,src}/arrow_tree_verify.*`) —
  один код для ctest-таргета и для нового `lsxforge --verify --input <model>`.
  Команда проверяет `<model>/logestix` (или сам logestix-каталог), печатает
  сводку и возвращает 1 при найденных проблемах; `LSX_ARROW_AUDIT_{FULL,
  MAX_BYTES}` действуют как в тесте. Проверено на реальном дереве Qwen3.5-9B
  (`files=68 footer_ok=68 deep=37 problems=0`).
- Gemma MoE/dense loader'ы переведены с `PinnedSchema::UnpackBatch` на
  `LoadTensorBatchViews`: dense-триплет больше не копируется при загрузке
  (A-07 закрыт). Страж `test_arrow_usage_guard` обновлён (эти файлы больше не
  в allowlist mmap/UnpackBatch).

## 4. Легаси-долг (не мигрирован в этой ветке)

`docs/OPEN_TASKS.md` + `ingesting-a-model` R1/R2 уже фиксируют паттерн
«все эксперты в pinned-арене». Аудит подтверждает его границы:

| семейство | host-хранение | цена |
|---|---|---|
| Qwen3.6 / Ornith | `MoEExpertStore` pinned slabs | ≈ дерево экспертов на диске (17 GiB) |
| Qwen3.8-Flash-Next | `LoadQwen4ExpExpertStore` pinned arena | 64 GiB + второй проход по диску |
| Mistral | `MoEExpertStore` pinned slabs | 61 GiB |
| GLM-5.3-Flash | pinned arena | ≈ дерево экспертов |
| Xing / flint / Hy | все эксперты резидентно на device | host-копия транзитна |

Это осознанное решение ветки: миграция — отдельная работа на 6+ семейств,
требующая бенчмарков до/после. Страж `test_arrow_usage_guard` фиксирует
вердикт каждого `UnpackBatch`-call-site, чтобы новый код не расширил долг
молча.

## 5. Валидация 26 деревьев

`test_arrow_tree_integrity` (footer-фаза для каждого файла + глубокая выборка
с бюджетом 256 MiB на дерево):

- деревьев: **26**;
- файлов `.arrow`: **273 254**, все открылись, у всех ровно один record batch
  и известная схема (`footer_ok = 273 254`);
- глубокая проверка: **1 693** файлов (offsets, границы буферов, NVFP4/raw
  инварианты);
- проблем: **0**;
- время прогона: ~211 с (footer-фаза с `CountRows` по всем файлам + выборка);
- лог: `/mnt/fast_storage/tmp/arrow_tree_audit3.log`.

Типы по дереву стабильны: `expert`, `dense-nvfp4`, `pinned`, `shared-expert`
(только Mistral/Kimi/DeepSeek-Pro), `tokenizer-vocab`, `tokenizer-merges`.
`budget_skips` — файлы, не попавшие в выборку (эксперты), а не ошибки.

## 6. Тесты

| target | label | что проверяет |
|---|---|---|
| `test_arrow_validation` | fast | 34 кейса: валидаторы, битые owning-batch'и (0/2 строки, offsets, тип, усечение), view-загрузчики, `.packed`/`.scales`, атомарная запись, `DecodeEntry`, бюджеты аллокаций записи |
| `test_arrow_usage_guard` | fast | R-A1 (нет `ReadableFile`), allowlist `MemoryMappedFile::Open`, таблица вердиктов `UnpackBatch`, отсутствие `MakeFileWriter` вне `schema.cpp`, матрица 29 моделей, запрет `ensure_alignment` |
| `test_arrow_tree_integrity` | convert (SKIP 4, TIMEOUT 1800) | все `.arrow` 26 деревьев: footer/схема + глубокая выборка |

## 6a. Прогоны

| набор | результат |
|---|---|
| `ctest --preset linux-debug -L fast` | **125/125** green (34 с) |
| `ctest --preset linux-debug -L fast -R "test_arrow_|test_schema_views|test_expert_store"` | green |
| loader-тесты inference-пресета (13 семейств) | **13/13** green |
| `test_arrow_tree_integrity` по 26 деревьям | green, 33 с |

Ранее красные `test_gemma_{moe,dense}_loader` и `test_gemma_{moe,dense}_model`
падали на этапе конвертера с `tokenizer: byte token missing in vocab: 'Ā'`.
Причина — не Arrow: эти тесты копировали generic `synthetic_tok.json` (0
байт-fallback кусков `<0xHH>`), тогда как `tokenizer_kind 11` (Gemma) требует
raw-text словарь с 256 `<0xHH>`; конвертерные gemma-тесты уже использовали
`synthetic_gemma_tok.json` после `b1c9c936`. Тесты переведены на тот же fixture.
Остаётся один красный во всём наборе — `test_gemma_dense_fsi_selfcheck`
(FSI-01): задокументированный parity-долг, `docs/SELFCHECK_MATRIX.md:34`
(FSI 2/3) и `docs/OPEN_TASKS.md` §R — «not a golden to rewrite»; рецепт,
закрывающий гейт, требует ~8 бит на каждый linear (33.7 GiB, не влезает в
карту).

## 7. Команды

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug -R "test_arrow_|test_schema_views|test_expert_store"
ctest --preset linux-debug -L fast
LSX_ARROW_AUDIT_ROOT=/mnt/fast_storage/huggingface \
  ctest --preset linux-debug -R test_arrow_tree_integrity --output-on-failure
# полная глубокая проверка (долго, читает payload всех экспертов):
LSX_ARROW_AUDIT_FULL=1 LSX_ARROW_AUDIT_MAX_BYTES=0 \
  ./build/linux-debug/tests/test_arrow_tree_integrity
```

## 8. Вне рамок

- миграция легаси-арен на нейтральный `ExpertStore` (A-08);
- переписывание safetensors-парсера конвертеров (не Arrow);
- изменение формата/схем на диске (формат стабилен);
- perf-оптимизация view-пути (A-07 оставлен как долг);
- Windows-прогон (remove-then-rename реализован, CI не запускался).
