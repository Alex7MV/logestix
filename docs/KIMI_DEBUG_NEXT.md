# Kimi-K2.7-Code: план продолжения отладки SelfCheck

Состояние: инференс проходит все 61 слой end-to-end (BF16 runtime,
коммит caea23f), 9/11 kimi-тестов зелёные. SelfCheck выполняется, но
top-1 неверный ('foss' вместо 'Paris').

## Следующий шаг: независимая валидация слоя 1

Все эталонные сверки layer-0 были "круговыми" (мой C++ против моего
numpy по моему же описанию семантики). Нужна проверка ПОЛНОГО слоя 1
(MoE) против оригинальных INT4 safetensors в torch.

1. C++: добавить env-gated дамп (`LSX_KIMI_DUMP_LAYER=1:<file>`)
   скрытого состояния после слоя N префилла в `KimiLayerStack`
   (`libs/lsxcommon/src/kimi_inference.cu`).
2. Torch (/tmp/opencode): свежий скрипт по HF-семантике kimi_k2
   (DeepSeekV2-стиль):
   - MLA: q_a_proj→q_a_layernorm→q_b_proj; kv_a_proj_with_m→kv_a_layernorm
     →kv_b_proj; yarn-rope(theta=50000, factor=64, beta_fast=32,
     beta_slow=128, orig_max_pos=4096); softmax_scale = mscale²/sqrt(192),
     mscale = 0.1*ln(factor)+1.
   - MoE: scores=sigmoid(gate(x)); top-8 по score+bias;
     w = renorm(sigmoid[top8]) * routed_scaling_factor(2.827);
     out = sum(w_i * expert_i) + shared_expert; routed experts из
     compressed-tensors INT4 group dequant.
3. Подать один и тот же вход (дамп после слоя 0), сравнить выход слоя 1.

## Зоны подозрения (в порядке вероятности)

- [ИСПРАВЛЕНО 2026-08-23] mscale: config задаёт mscale=mscale_all_dim=1.0
  -> attention factor = 1.0, KimiAttnScale = 1/sqrt(192) без буста.
  beta_slow=1.0 уже был верен в BuildYarnInvFreq.
  Top-1 НЕ изменился ('foss') -> основная ошибка в другом.
- применение kv_b_proj и восстановление q_nope через absorption
  (проверить знак/порядок: HF: q_nope = q_b[..., :nope] проходит через
  latent; scores_nope = einsum(q_nope_absorbed, k_latent))
- renorm весов роутера до/после scaling (HF: w = w/sum * scaling)
- позиционные смещения rope между prefill/decode
- СЛЕДУЮЩИЙ ШАГ остаётся: дамп после слоя N + свежий torch-референс
  слоя 1 из оригинальных INT4 safetensors

## Запуск

ctest --preset linux-debug -R kimi
./build/linux-debug/tests/test_kimi_selfcheck
