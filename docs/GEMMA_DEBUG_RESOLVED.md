# Gemma-4-26B-A4B-it: SelfCheck решён (2026-08-24)

SelfCheck проходит: 'The capital of France is' → «The capital of France is
**Paris**.» (жадное продолжение этой модели; 'Paris' в первых токенах),
бит-точная детерминированность по циклам.

## Root causes (по порядку находки)

1. **Фрейминг неполный**: после `<|turn>model\n` шаблон (enable_thinking
   default=false) добавляет ПУСТОЙ thinking-канал
   `<|channel>thought\n<channel|>` (=100, 45518, 107, 101). Без него модель
   предсказывает мусор. Фрейминг при этом стал **модель-зависимым**:
   `IModel::BuildPromptInputIds` (виртуальный, дефолт = сырой энкод), GLM и
   Gemma владеют своими дескрипторами (`glm::kFraming` / `gemma::kFraming`),
   центрального реестра больше нет.

2. **Двойной BOS**: энкодер сам добавляет bos_id внутрь промпта + SelfCheck
   вставлял второй BOS в начало. Теперь политика BOS — часть дескриптора
   фрейминга (leading_bos): авто-BOS переносится на позицию 0.

3. **layer_scalar / router.per_expert_scale терялись**: конвертер хранит их
   BF16 (2 байта/значение), а лоадер проверял размер по F32 → тензоры молча
   пропускались, layer_scalar не применялся (скрытые состояния разрастались в
   ~15 раз). Фикс: размерные гарды по dtype (+ регрессия в фикстуре/тесте).

4. **Attention scale**: HF Gemma4TextAttention использует scaling = 1.0
   (q/k нормированы per-head); был 1/sqrt(head_dim).

5. **__constant__ kE2M1Table = мусор в новом TU**: заголовочный __constant__
   массив при линковке нескольких object-библиотек дал битую копию в
   lsxcommon_gemma_cu (RTX 5090 / CUDA 13.3). Заменён на inline-функцию
   `nvfp4::E2M1Value(nib)` с иммедиантами — глобального хранилища больше нет,
   ODR-алиасинг невозможен.

6. **cudaEventElapsedTime на cudaEventDisableTiming-событиях** возвращал
   sticky "invalid resource handle"; первый же getLastError в decode-цикле
   трактовал его как ошибку генерации → обрыв после первого токена. TTFT/total
   переведены на std::chrono.

## Верификация

- Абсолютный эталон: реальный HF transformers 5.15.1 (CPU, bf16) на том же
  framed-промпте — top-1 'The', 'Paris' топ-2, greedy =
  «The capital of France is **…»; логиты C++ vs HF corr=0.985, топ-10
  пересекается 8/10.
- Suite: gemma 9/9, лёгкие 53/53, glm/kimi/deepseek/integrated/blackwell
  26/26. Прешединговые падения test_tokenizer (FIM SP-нормализация) — отдельный
  баг ветки, к этим правкам не относится.

## Методика

Env-gated дампы скрытых состояний по слоям/стадиям + свежий torch-референс из
оригинальных safetensors (BF16) + решающая сверка с настоящим HF forward как
ground truth.
