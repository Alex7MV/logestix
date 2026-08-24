# Kimi-K2.7-Code: план продолжения отладки SelfCheck

РЕШЕНО 2026-08-24. SelfCheck проходит: 'The capital of France is' → 'Paris'
(top-1 tok=17374), 3+ циклов с бит-точной детерминированностью final_logits.

## Root causes (найдены независимой валидацией слоя 1)

1. **INT4 zero-point convention** (`kimi_int4_dequant_kernel`): compressed-tensors
   pack-quantized symmetric int4 хранит unsigned нимбл с неявным zero-point 8
   (offset binary), а ядро декодировало как two's complement (`nib>7 ? nib-16 :
   nib`). Правильно: `(nib - 8) * scale`. Диагностика: гистограмма нимблов
   исходного чекпоинта симметрична вокруг q=8; group-mean статистика ~0 только
   при offset-binary; corr(NVFP4, INT4 twos-comp)=0.994 подтверждает, что
   конвертер закодировал неверные веса. Юнит-тест кодировал ту же ошибку
   (круговая проверка) — переписан на offset-binary. Модель переконвертирована.

2. **Недетерминированный float atomicAdd** в трёх RMSNorm-ядрах
   (`kimi_rmsnorm_rows_kernel`, `kimi_rmsnorm_rows_op_kernel`,
   `kimi_kv_postprocess_kernel`): порядок суммирования варпов менялся между
   запусками → биты final_logits гуляли. Заменено на детерминированную редукцию:
   partials в shared memory + фиксированный последовательный сумминг thread 0.

## Что НЕ было причиной

- yarn attention scale: softmax-scale = 1/sqrt(192) верен для этого конфига
  (mscale=mscale_all_dim=1.0); HF DeepseekV3Attention применяет mscale²-буст
  через yarn_get_mscale(factor, mscale_all_dim) — но это отдельная величина от
  rotary attention_factor; проверено численно слоем 1 (attention residual
  сходится с max|d|=0.0017).
- Absorption kv_b_proj, renorm роутера (w = sigmoid/sum * 2.827 — совпадает с
  HF до 1e-5), позиционные смещения rope, shared_experts (вес 1.0) — всё
  сверено поэлементно против свежего torch-референса из оригинальных INT4.

## Методика (для следующих моделей)

Env-gated дампы скрытого состояния после слоя N префилла + свежий torch-референс
слоя из оригинальных safetensors (не своих конвертированных файлов) локализуют
расхождение за минуты: attention residual → MoE → shared/routed → router.
