// Формулы и справочники калькулятора. Всё в одном объекте LLMCalc, без DOM.
var LLMCalc = (() => {
  const GB = 1024 ** 3;
  const BW_EFF = 0.7;       // реальная доля пропускной способности, которую достигают llama.cpp / MLX / vLLM при генерации
  const COMPUTE_EFF = 0.5;  // реальная доля пиковых TFLOPS при обработке промпта
  const RUNTIME_BYTES = 0.5 * GB;   // сама программа, драйвер, CUDA-контекст
  const OS_RESERVE_SYS = 0.85;      // сколько системной ОЗУ можно занять под выгрузку

  // Реальные биты на вес у GGUF-квантов (с учётом масштабов и смешанных типов слоёв)
  // fmt — в каких оболочках квант родной: gguf (llama.cpp и производные), mlx (Mac), fp4 (vLLM, Strata на NVIDIA), gptq (vLLM), any — везде
  const QUANTS = {
    FP16:    { bpw: 16,   fmt: 'any',  label: 'FP16 / BF16', note: 'без потерь, вдвое больше Q8' },
    Q8_0:    { bpw: 8.5,  fmt: 'gguf', label: 'Q8_0',        note: 'почти без потерь' },
    MLX8:    { bpw: 8.5,  fmt: 'mlx',  label: 'MLX 8-bit',   note: 'почти без потерь; формат MLX (группы по 64)' },
    Q6_K:    { bpw: 6.56, fmt: 'gguf', label: 'Q6_K',        note: 'потери незаметны' },
    Q5_K_M:  { bpw: 5.69, fmt: 'gguf', label: 'Q5_K_M',      note: 'лёгкие потери' },
    Q4_K_M:  { bpw: 4.85, fmt: 'gguf', label: 'Q4_K_M',      note: 'стандарт: баланс размера и качества' },
    UD_Q4:   { bpw: 4.6,  fmt: 'gguf', label: 'UD-Q4_K_XL',  note: 'динамический квант Unsloth: качество Q4_K_M, файл чуть меньше' },
    MLX4:    { bpw: 4.5,  fmt: 'mlx',  label: 'MLX 4-bit',   note: 'стандарт для Mac (группы по 64: 4 бита + масштаб и сдвиг)' },
    NVFP4:   { bpw: 4.5,  fmt: 'fp4',  label: 'NVFP4',       note: 'аппаратный FP4 у Blackwell (RTX 50, RTX PRO, GB10)' },
    Q4_0:    { bpw: 4.55, fmt: 'gguf', label: 'Q4_0',        note: 'быстрый на Apple, чуть хуже K_M' },
    MXFP4:   { bpw: 4.25, fmt: 'any',  label: 'MXFP4',       note: 'родной формат GPT-OSS' },
    GPTQ4:   { bpw: 4.25, fmt: 'gptq', label: 'AWQ / GPTQ int4', note: 'для vLLM и SGLang' },
    IQ4_XS:  { bpw: 4.25, fmt: 'gguf', label: 'IQ4_XS',      note: 'почти как Q4_K_M, на 12 % меньше' },
    Q3_K_M:  { bpw: 3.91, fmt: 'gguf', label: 'Q3_K_M',      note: 'заметные потери' },
    Q2_K:    { bpw: 3.35, fmt: 'gguf', label: 'Q2_K',        note: 'сильные потери' },
    IQ3_XXS: { bpw: 3.06, fmt: 'gguf', label: 'IQ3_XXS',     note: 'для больших MoE, у которых иначе не влезает' },
    IQ2_XXS: { bpw: 2.06, fmt: 'gguf', label: 'IQ2_XXS',     note: 'крайний случай, качество падает' },
  };
  // KV-кэш: q8_0 и q4_0 в llama.cpp хранят масштаб на блок из 32 — реально 8,5 и 4,5 бита; fp8 и NVFP4 в vLLM — столько же
  const KV_QUANTS = { FP16: 16, Q8_0: 8.5, Q4_0: 4.5, TQ3: 3.25 };
  const KV_LABELS = { FP16: 'FP16', Q8_0: 'Q8_0 / FP8', Q4_0: 'Q4_0 / NVFP4', TQ3: 'TurboQuant 3 бит (vLLM)' };

  // Модели: params/active в штуках параметров; kvFactor 1 — MLA (один латентный вектор вместо пары K/V);
  // window/localRatio — скользящее окно: доля слоёв с локальным вниманием и его размер;
  // kvLayers — сколько слоёв вообще держат KV (гибриды с линейным вниманием / Mamba); kvScale — сжатие кэша.
  // Источник чисел для моделей 2026 года — config.json на Hugging Face (проверено 13.09.2026).
  const B = 1e9;
  const MODELS = [];
  // m(семейство, имя, всего млрд, активных млрд, слои, KV-головы, размер головы, скрытая, макс. контекст, доп.)
  const m = (family, name, params, active, layers, kvHeads, headDim, hidden, maxCtx, extra = {}) =>
    MODELS.push({ family, name, params: params * B, active: active * B, layers, kvHeads, headDim, hidden, maxCtx, ...extra });
  const SW = (window, localRatio) => ({ window, localRatio });   // скользящее окно: размер и доля локальных слоёв
  const MLA = { kvHeads: 1, headDim: 576, kvFactor: 1 };          // DeepSeek-подобное сжатое внимание: один вектор 576 на слой

  // --- Meta Llama
  m('Llama', 'Llama 2 7B',                7,    7,    32, 32, 128, 4096,  4096);
  m('Llama', 'Llama 2 13B',               13,   13,   40, 40, 128, 5120,  4096);
  m('Llama', 'Llama 2 70B',               69,   69,   80, 8,  128, 8192,  4096);
  m('Llama', 'CodeLlama 34B',             34,   34,   48, 8,  128, 8192,  16384);
  m('Llama', 'Llama 3.2 1B',              1.24, 1.24, 16, 8,  64,  2048,  131072);
  m('Llama', 'Llama 3.2 3B',              3.21, 3.21, 28, 8,  128, 3072,  131072);
  m('Llama', 'Llama 3.1 8B / R1-Distill-8B', 8.03, 8.03, 32, 8, 128, 4096, 131072);
  m('Llama', 'Llama 3.3 70B / 3.1 70B / R1-Distill-70B', 70.6, 70.6, 80, 8, 128, 8192, 131072);
  m('Llama', 'Llama 3.1 405B',            405,  405,  126, 8, 128, 16384, 131072);
  m('Llama', 'Llama 4 Scout 109B-A17B (MoE)',    109, 17, 48, 8, 128, 5120, 1048576, SW(8192, 3/4));
  m('Llama', 'Llama 4 Maverick 400B-A17B (MoE)', 400, 17, 48, 8, 128, 5120, 1048576, SW(8192, 3/4));
  // --- Qwen
  m('Qwen', 'Qwen2.5 0.5B',               0.49, 0.49, 24, 2,  64,  896,   32768);
  m('Qwen', 'Qwen2.5 1.5B / R1-Distill-1.5B', 1.54, 1.54, 28, 2, 128, 1536, 32768);
  m('Qwen', 'Qwen2.5 3B',                 3.09, 3.09, 36, 2,  128, 2048,  32768);
  m('Qwen', 'Qwen2.5 7B / Coder 7B / R1-Distill-7B', 7.6, 7.6, 28, 4, 128, 3584, 131072);
  m('Qwen', 'Qwen2.5 14B / R1-Distill-14B', 14.8, 14.8, 48, 8, 128, 5120, 131072);
  m('Qwen', 'Qwen2.5 32B / Coder 32B / QwQ / R1-Distill-32B', 32.8, 32.8, 64, 8, 128, 5120, 131072);
  m('Qwen', 'Qwen2.5 72B',                72.7, 72.7, 80, 8,  128, 8192,  131072);
  m('Qwen', 'Qwen3 0.6B',                 0.6,  0.6,  28, 8,  128, 1024,  32768);
  m('Qwen', 'Qwen3 1.7B',                 1.7,  1.7,  28, 8,  128, 2048,  32768);
  m('Qwen', 'Qwen3 4B',                   4.0,  4.0,  36, 8,  128, 2560,  131072);
  m('Qwen', 'Qwen3 8B',                   8.2,  8.2,  36, 8,  128, 4096,  131072);
  m('Qwen', 'Qwen3 14B',                  14.8, 14.8, 40, 8,  128, 5120,  131072);
  m('Qwen', 'Qwen3 32B',                  32.8, 32.8, 64, 8,  128, 5120,  131072);
  m('Qwen', 'Qwen3 30B-A3B / Coder 30B-A3B (MoE)', 30.5, 3.3, 48, 4, 128, 2048, 262144);
  m('Qwen', 'Qwen3 235B-A22B (MoE)',      235,  22,   94, 4,  128, 4096,  262144);
  m('Qwen', 'Qwen3-Coder 480B-A35B (MoE)', 480, 35,   62, 8,  128, 6144,  262144);
  m('Qwen', 'Qwen3-Next 80B-A3B (MoE, гибрид)', 80, 3, 48, 2, 256, 2048, 262144, { kvLayers: 12 });
  // Qwen 3.5 / 3.6 / 3.8 — гибрид Gated DeltaNet + полное внимание в каждом 4-м слое (config.json, 2026)
  m('Qwen 3.5–3.8', 'Qwen3.5 0.8B',                 0.8,  0.8,  24, 2, 256, 1024, 262144, { kvLayers: 6 });
  m('Qwen 3.5–3.8', 'Qwen3.5 2B',                   2,    2,    24, 2, 256, 2048, 262144, { kvLayers: 6 });
  m('Qwen 3.5–3.8', 'Qwen3.5 4B',                   4,    4,    32, 4, 256, 2560, 262144, { kvLayers: 8 });
  m('Qwen 3.5–3.8', 'Qwen3.5 9B',                   9,    9,    32, 4, 256, 4096, 262144, { kvLayers: 8 });
  m('Qwen 3.5–3.8', 'Qwen3.5 27B / Qwen3.6 27B',    27,   27,   64, 4, 256, 5120, 262144, { kvLayers: 16 });
  m('Qwen 3.5–3.8', 'Qwen3.5 35B-A3B / Qwen3.6 35B-A3B (MoE)', 35, 3, 40, 2, 256, 2048, 262144, { kvLayers: 10 });
  m('Qwen 3.5–3.8', 'Qwen3.5 122B-A10B (MoE)',      122,  10,   48, 2, 256, 3072, 262144, { kvLayers: 12 });
  m('Qwen 3.5–3.8', 'Qwen3.5 397B-A17B (MoE)',      397,  17,   60, 2, 256, 4096, 262144, { kvLayers: 15 });
  m('Qwen 3.5–3.8', 'Qwen3.8 27B',                  27,   27,   64, 4, 256, 5120, 262144, { kvLayers: 16 });
  m('Qwen 3.5–3.8', 'Qwen3.8-Max 2.4T-A95B (MoE)',      2400, 95,   92, 4, 256, 8192, 262144, { kvLayers: 23 });
  // Flash-Next: 125B трансформер + 51B n-gram-таблиц + 4B MTP-голова. llama.cpp грузит все ~176B;
  // MLX/MTPLX и Strata держат n-gram-таблицы на SSD (ssdParams), тогда в памяти ~125B.
  m('Qwen 3.5–3.8', 'Qwen3.8-Flash-Next 176B-A6B (MoE, n-gram 51B)', 176, 6, 48, 2, 256, 2560, 262144, { kvLayers: 12, ssdParams: 51 * B, note: 'из них 51B — n-gram-таблицы' });
  // --- Google Gemma
  m('Gemma', 'Gemma 2 2B',                2.6,  2.6,  26, 4,  256, 2304,  8192,  SW(4096, 1/2));
  m('Gemma', 'Gemma 2 9B',                9.2,  9.2,  42, 8,  256, 3584,  8192,  SW(4096, 1/2));
  m('Gemma', 'Gemma 2 27B',               27.2, 27.2, 46, 16, 128, 4608,  8192,  SW(4096, 1/2));
  m('Gemma', 'Gemma 3 1B',                1.0,  1.0,  26, 1,  256, 1152,  32768, SW(512, 5/6));
  m('Gemma', 'Gemma 3 4B',                4.3,  4.3,  34, 4,  256, 2560,  131072, SW(1024, 5/6));
  m('Gemma', 'Gemma 3 12B',               12.2, 12.2, 48, 8,  256, 3840,  131072, SW(1024, 5/6));
  m('Gemma', 'Gemma 3 27B',               27.4, 27.4, 62, 16, 128, 5376,  131072, SW(1024, 5/6));
  m('Gemma', 'Gemma 3n E4B',              8.0,  4.0,  35, 2,  256, 2048,  32768, SW(512, 4/5));
  // Gemma 4 (2026): часть слоёв делит KV с предыдущими (num_kv_shared_layers), у E-моделей — per-layer embeddings
  m('Gemma', 'Gemma 4 E2B',               5.1,  2.3,  35, 1,  256, 1536,  131072, { ...SW(512, 29/35), kvLayers: 15 });
  m('Gemma', 'Gemma 4 E4B',               8.0,  4.5,  42, 2,  256, 2560,  131072, { ...SW(512, 6/7), kvLayers: 24 });
  m('Gemma', 'Gemma 4 12B',               12,   12,   48, 8,  256, 3840,  131072, SW(1024, 5/6));
  m('Gemma', 'Gemma 4 26B-A4B (MoE)',     26,   3.8,  30, 8,  256, 2816,  262144, SW(1024, 5/6));
  m('Gemma', 'Gemma 4 31B',               31,   31,   60, 16, 256, 5376,  262144, SW(1024, 5/6));
  // --- Mistral
  m('Mistral', 'Mistral 7B v0.3',         7.25, 7.25, 32, 8,  128, 4096,  32768);
  m('Mistral', 'Ministral 8B (2024)',     8.0,  8.0,  36, 8,  128, 4096,  131072);
  m('Mistral', 'Ministral 3 3B',          3.4,  3.4,  26, 8,  128, 3072,  262144);
  m('Mistral', 'Ministral 3 8B',          8.4,  8.4,  34, 8,  128, 4096,  262144);
  m('Mistral', 'Ministral 3 14B',         14.4, 14.4, 40, 8,  128, 5120,  262144);
  m('Mistral', 'Mistral Nemo 12B',        12.2, 12.2, 40, 8,  128, 5120,  131072);
  m('Mistral', 'Codestral 22B',           22.2, 22.2, 56, 8,  128, 6144,  32768);
  m('Mistral', 'Mistral Small 3.x / Magistral / Devstral Small 2 24B', 24, 24, 40, 8, 128, 5120, 131072);
  m('Mistral', 'Mistral Small 4 119B-A6B (MoE, MLA)', 119, 6.5, 36, 1, 320, 4096, 1048576, { kvFactor: 1, note: 'сжатое внимание 256+64' });
  m('Mistral', 'Mistral Medium 3.5 128B', 128,  128,  88, 8,  128, 12288, 262144);
  m('Mistral', 'Mixtral 8x7B (MoE)',      46.7, 12.9, 32, 8,  128, 4096,  32768);
  m('Mistral', 'Mixtral 8x22B (MoE)',     141,  39,   56, 8,  128, 6144,  65536);
  m('Mistral', 'Mistral Large 2 123B',    123,  123,  88, 8,  128, 12288, 131072);
  m('Mistral', 'Mistral Large 3 675B-A41B (MoE)', 675, 41, 60, 8, 128, 7168, 262144, { note: 'слои и KV-головы оценены, конфиг закрыт' });
  m('Mistral', 'Mistral Large 4 1T-A52B (MoE, MLA)', 1050, 52, 72, 1, 576, 6144, 1048576, { ...MLA, note: 'индексатор читает часть кэша, хранится весь' });
  // --- DeepSeek
  m('DeepSeek', 'DeepSeek-V2-Lite 16B-A2.4B (MoE, MLA)', 15.7, 2.4, 27, 1, 576, 2048, 32768, MLA);
  m('DeepSeek', 'DeepSeek-V2 / Coder-V2 236B-A21B (MoE, MLA)', 236, 21, 60, 1, 576, 5120, 131072, MLA);
  m('DeepSeek', 'DeepSeek V3 / V3.1 / V3.2 / R1 671B-A37B (MoE, MLA)', 671, 37, 61, 1, 576, 7168, 131072, MLA);
  // V4 (2026): сжатое разрежённое внимание, по карточке модели KV ≈ 10 % от V3.2
  m('DeepSeek', 'DeepSeek V4 Flash 284B-A13B (MoE)', 284, 13, 43, 1, 576, 4096, 1048576, { ...MLA, kvScale: 0.1, note: 'KV ≈ 10 % от V3 по карточке' });
  m('DeepSeek', 'DeepSeek V4 Pro 1.6T-A49B (MoE)',   1600, 49, 61, 1, 576, 7168, 1048576, { ...MLA, kvScale: 0.1, note: 'KV ≈ 10 % от V3 по карточке' });
  m('DeepSeek', 'DeepSeek V4.1 Flash 763B-A16B (MoE, Engram)', 763, 16, 40, 1, 576, 5120, 1048576, { ...MLA, kvScale: 0.0125, ssdParams: 196 * B, note: 'из них 196B — Engram-память; KV ≈ 1/8 от V4 Flash по карточке' });
  // --- OpenAI
  m('OpenAI', 'GPT-OSS 20B (MoE)',        21,   3.6,  24, 8,  64,  2880,  131072, SW(128, 1/2));
  m('OpenAI', 'GPT-OSS 120B (MoE)',       117,  5.1,  36, 8,  64,  2880,  131072, SW(128, 1/2));
  // --- Microsoft Phi
  m('Phi', 'Phi-3 / 3.5 mini 3.8B',       3.8,  3.8,  32, 32, 96,  3072,  131072);
  m('Phi', 'Phi-4 mini 3.8B',             3.8,  3.8,  32, 8,  128, 3072,  131072);
  m('Phi', 'Phi-3 medium 14B',            14,   14,   40, 10, 128, 5120,  131072);
  m('Phi', 'Phi-4 / Phi-4 reasoning 14B', 14.7, 14.7, 40, 10, 128, 5120,  16384);
  // --- Zhipu GLM
  m('GLM', 'GLM-4 9B',                    9.4,  9.4,  40, 2,  128, 4096,  131072);
  m('GLM', 'GLM-4.5 Air 106B-A12B (MoE)', 106,  12,   46, 8,  128, 4096,  131072);
  m('GLM', 'GLM-4.5 / 4.6 355B-A32B (MoE)', 355, 32,  92, 8,  128, 5120,  131072);
  m('GLM', 'GLM-5 / 5.1 / 5.2 / 5.3 744B-A40B (MoE, MLA)', 744, 40, 78, 1, 576, 6144, 1048576, MLA);
  m('GLM', 'GLM-5.3-Flash 320B-A18B (MoE, гибрид, MLA)', 320, 18, 45, 1, 512, 4096, 1048576, { kvFactor: 1, kvLayers: 11 });
  // --- Moonshot Kimi
  m('Kimi', 'Kimi Linear 48B-A3B (MoE, гибрид)', 48, 3, 27, 1, 576, 2304, 1048576, { ...MLA, kvLayers: 7 });
  m('Kimi', 'Kimi K2 / K2 Thinking 1T-A32B (MoE, MLA)', 1026, 32, 61, 1, 576, 7168, 131072, MLA);
  m('Kimi', 'Kimi K2.5 / K2.6 / K2.7 Code 1T-A32B (MoE, MLA)', 1026, 32, 61, 1, 576, 7168, 262144, MLA);
  m('Kimi', 'Kimi K3 2.8T-A104B (MoE, гибрид)', 2800, 104, 93, 1, 576, 7168, 1048576, { ...MLA, kvLayers: 24 });
  // --- MiniMax, Xiaomi, Tencent, Baidu, ByteDance, Ant
  m('Китайские MoE', 'MiniMax-M1 456B-A46B (MoE, гибрид)', 456, 46, 80, 8, 128, 6144, 1048576, { kvLayers: 10 });
  m('Китайские MoE', 'MiniMax-M2 230B-A10B (MoE)', 230, 10, 62, 8, 128, 3072, 196608);
  m('Китайские MoE', 'MiniMax-M2.5 / M2.7 229B-A10B (MoE)', 229, 10, 62, 8, 128, 3072, 204800);
  m('Китайские MoE', 'MiniMax-M3 428B-A23B (MoE)', 428, 23, 60, 4, 128, 6144, 1048576, { note: 'разрежённое внимание, кэш хранится целиком' });
  m('Китайские MoE', 'MiniMax-M3.1-Flash 428B-A23B (MoE)', 428, 23, 60, 4, 128, 6144, 1048576, { note: 'блочно-разрежённое внимание, кэш хранится целиком' });
  m('Китайские MoE', 'MiMo-V2-Flash 309B-A15B (MoE)', 309, 15, 48, 4, 192, 4096, 262144, SW(128, 39/48));
  m('Китайские MoE', 'MiMo-V2.5 310B-A15B (MoE)', 310, 15, 48, 4, 192, 4096, 1048576, SW(128, 39/48));
  m('Китайские MoE', 'MiMo-V2.5-Pro 1T-A42B (MoE)', 1020, 42, 70, 8, 192, 6144, 1048576, SW(128, 6/7));
  m('Китайские MoE', 'MiMo-V2.6-Flash 310B-A15B (MoE)', 310, 15, 48, 4, 160, 4096, 1048576, { ...SW(128, 39/48), note: 'K 192 + V 128' });
  m('Китайские MoE', 'MiMo-V2.6-Pro 1T-A42B (MoE)', 1024, 42, 70, 8, 160, 6144, 1048576, { ...SW(128, 6/7), note: 'K 192 + V 128' });
  m('Китайские MoE', 'Ring-2.6-1T 1T-A63B (MoE, MLA)', 1000, 63, 80, 1, 576, 8192, 131072, MLA);
  m('Китайские MoE', 'Hunyuan-A13B 80B (MoE)', 80, 13, 32, 8, 128, 4096, 262144);
  m('Китайские MoE', 'ERNIE 4.5 21B-A3B (MoE)', 21, 3, 28, 4, 128, 2560, 131072);
  m('Китайские MoE', 'ERNIE 4.5 300B-A47B (MoE)', 300, 47, 54, 8, 128, 8192, 131072);
  m('Китайские MoE', 'Seed-OSS 36B',      36,   36,   64, 8,  128, 5120,  524288);
  m('Китайские MoE', 'Step 3.5 Flash 196B-A11B (MoE)', 196, 11, 45, 8, 128, 4096, 262144, SW(512, 3/4));
  m('Китайские MoE', 'Step 5 Preview 600B-A27B (MoE)', 600, 27, 92, 4, 192, 4096, 1048576, { ...SW(512, 3/4), note: 'полные слои — разрежённое внимание top-512' });
  // --- NVIDIA Nemotron 3 (гибрид Mamba-2 + MoE, внимание лишь в части слоёв)
  m('NVIDIA', 'Nemotron 3 Nano 30B-A3B',  31.6, 3.5,  52, 2,  128, 2688,  262144, { kvLayers: 6 });
  m('NVIDIA', 'Nemotron 3 Super 120B-A12B', 120, 12,  88, 2,  128, 4096,  262144, { kvLayers: 9 });
  m('NVIDIA', 'Nemotron 3 Ultra 550B-A55B', 550, 55,  128, 2, 128, 8192,  262144, { kvLayers: 16 });
  m('NVIDIA', 'Nemotron 3.5 Lightning 30B-A3B', 31.6, 3, 52, 2, 128, 2688, 262144, { kvLayers: 6 });
  m('NVIDIA', 'Nemotron 3.5 Bolt 74B-A9B',  74.9, 9,  88, 2,  128, 4096,  262144, { kvLayers: 8 });
  m('NVIDIA', 'Nemotron 3.5 Super VL 120B-A12B', 124.5, 12, 88, 2, 128, 4096, 1048576, { kvLayers: 8 });
  // Dory: 36 блоков внимания, середина (20 из них) проходится дважды со своим кэшем на каждый проход → 56 слоёв KV
  m('NVIDIA', 'Nemotron-Labs Dory 4B (петля)', 4, 4, 56, 4, 256, 2560, 262144, { ...SW(1024, 3/4), note: 'середина модели идёт дважды: генерация медленнее оценки примерно на треть' });
  // --- Poolside
  m('Poolside', 'Laguna XS 2.1 33B-A3B (MoE)', 33, 3, 40, 8, 128, 2048, 262144, SW(512, 3/4));
  m('Poolside', 'Laguna S 2.1 118B-A8B (MoE)', 118, 8, 48, 8, 128, 3072, 1048576, SW(512, 3/4));
  // --- Meta Muse, Reflection, Marin (открытые веса, осень 2026)
  m('Прочие MoE', 'Meta Muse Spark Open 816B-A42B (MoE)', 822, 42, 62, 16, 64, 8192, 1048576, SW(2048, 46/62));
  m('Прочие MoE', 'Reflection Beam 1.0 501B-A23B (MoE)', 501, 23, 52, 8, 128, 5120, 1048576, SW(512, 43/52));
  m('Прочие MoE', 'Marin Snowball 67B-A2B v0 (MoE)', 67, 2, 26, 5, 128, 2560, 262144, SW(2048, 3/4));
  m('Прочие MoE', 'Ling-3.0-flash-VL 124B-A5.5B (MoE, гибрид, MLA)', 124, 5.5, 42, 1, 576, 2560, 131072, { ...MLA, kvLayers: 7 });
  // --- Cohere
  m('Cohere', 'Command R 35B',            35,   35,   40, 8,  128, 8192,  131072);
  m('Cohere', 'Command R+ 104B',          104,  104,  64, 8,  128, 12288, 131072);
  m('Cohere', 'Command A 111B',           111,  111,  64, 8,  128, 12288, 262144, SW(4096, 3/4));
  // --- Прочие открытые
  m('Прочие', 'TinyLlama 1.1B',           1.1,  1.1,  22, 4,  64,  2048,  2048);
  m('Прочие', 'SmolLM3 3B',               3.1,  3.1,  36, 4,  128, 2048,  65536);
  m('Прочие', 'LFM3 1.7B (гибрид со свёртками)', 1.7, 1.7, 24, 4, 128, 2048, 262144, { kvLayers: 11, ...SW(256, 5/11) });
  m('Прочие', 'Granite 3.x 8B',           8.2,  8.2,  40, 8,  128, 4096,  131072);
  m('Прочие', 'Granite 4.0 H Small 32B-A9B (MoE, гибрид)', 32, 9, 40, 8, 128, 4096, 131072, { kvLayers: 10 });
  m('Прочие', 'OLMo 2 7B',                7.3,  7.3,  32, 32, 128, 4096,  4096);
  m('Прочие', 'OLMo 2 13B',               13.7, 13.7, 40, 40, 128, 5120,  4096);
  m('Прочие', 'OLMo 2 32B',               32.2, 32.2, 64, 40, 128, 5120,  4096);
  m('Прочие', 'OLMo 3 7B',                7.3,  7.3,  32, 32, 128, 4096,  65536, SW(4096, 3/4));
  m('Прочие', 'OLMo 3 32B',               32.2, 32.2, 64, 8,  128, 5120,  65536, SW(4096, 3/4));
  m('Прочие', 'Trinity Mini 26B-A3B (MoE)', 26, 3,    32, 4,  128, 2048,  131072, SW(2048, 3/4));
  m('Прочие', 'Yi 1.5 34B',               34.4, 34.4, 60, 8,  128, 7168,  32768);
  m('Прочие', 'StarCoder2 15B',           16,   16,   40, 4,  128, 6144,  16384);
  m('Прочие', 'DBRX 132B-A36B (MoE)',     132,  36,   40, 8,  128, 6144,  32768);
  m('Прочие', 'EXAONE 4.0 32B',           32,   32,   64, 8,  128, 5120,  131072, SW(4096, 3/4));
  m('Прочие', 'Falcon3 10B',              10.3, 10.3, 40, 4,  256, 3072,  32768);

  // MTP-голова в config.json (num_nextn_predict_layers, mtp_num_hidden_layers, num_mtp_modules) — проверено 01.11.2026
  const MTP_PREFIXES = ['Qwen3.5 ', 'Qwen3.8 27B', 'Qwen3.8-Flash-Next', 'DeepSeek V3', 'DeepSeek V4', 'GLM-4.5', 'GLM-5',
    'MiniMax-M2', 'MiniMax-M3 ', 'MiMo-V2.6-Flash', 'Step 3.5', 'Step 5', 'Nemotron 3 Super', 'Nemotron 3 Ultra',
    'Nemotron 3.5 Lightning', 'Nemotron 3.5 Super VL'];
  // Модели, под которые написан Strata (README, docs/MODELS.md, 31.10.2026)
  const STRATA_PREFIXES = ['Qwen3.8-Flash-Next', 'Qwen3.5 27B / Qwen3.6', 'Qwen3.5 35B-A3B / Qwen3.6', 'DeepSeek V4.1 Flash'];
  for (const md of MODELS) {
    if (MTP_PREFIXES.some(p => md.name.startsWith(p))) md.mtp = true;
    if (STRATA_PREFIXES.some(p => md.name.startsWith(p))) md.strata = true;
  }

  // Железо. memGB — память ускорителя (или вся ОЗУ при unified). tflops — практичные FP16 без разрежённости
  // (у NVIDIA это половина маркетинговых tensor-цифр, у Apple M1–M4 — пиковые FP32; у M5/M6 с нейроускорителями в GPU
  // взято ×2 к M4-аналогу, консервативно к заявленным ×3–4). Для дискретных карт sysMemGB/sysBandwidthGBs — ОЗУ под выгрузку.
  // Коробки и Mac проверены по спецификациям производителей 13.09.2026.
  const HARDWARE = [];
  // vendor — по группе: от него зависит, какие оболочки запускаются; ssdGBs — чтение с SSD при подкачке экспертов
  const vendorOf = (group, name) => /Apple/.test(group) ? 'apple' : /GB10|Jetson|^NVIDIA/.test(group) ? 'nvidia'
    : /Strix|^AMD/.test(group) ? 'amd' : /^Intel/.test(group) ? 'intel' : /процессор/.test(group) ? 'cpu'
    : /Intel/.test(name) ? 'intel' : /AMD/.test(name) ? 'amd' : 'qualcomm';
  const add = (group, name, memGB, bandwidthGBs, tflops, opts = {}) => {
    const vendor = vendorOf(group, name);
    HARDWARE.push({ group, name, memGB, bandwidthGBs, tflops, unified: false, sysMemGB: 32, sysBandwidthGBs: 80, sysTflops: 1,
      vendor, ssdGBs: vendor === 'apple' ? 12 : 6, ...opts });
  };
  const uni = (group, name, bw, tf, mems, def) => add(group, name, def ?? mems[Math.floor(mems.length / 2)], bw, tf, { unified: true, memOptions: mems });

  // --- Apple: что продаётся сейчас (сентябрь 2026)
  const G_MAC = 'Apple · текущие Mac';
  uni(G_MAC, 'Mac mini · M6 (12-core GPU)',                170,  13,  [16, 24, 32]);
  uni(G_MAC, 'Mac mini · M5 Pro (16/20-core GPU)',         307,  20,  [24, 48, 64]);
  uni(G_MAC, 'Mac Studio · M5 Max 32-core GPU',            460,  32,  [36, 48, 64, 128]);
  uni(G_MAC, 'Mac Studio · M5 Max 40-core GPU',            614,  40,  [36, 48, 64, 128], 128);
  uni(G_MAC, 'Mac Studio · M5 Ultra 64-core GPU',          1200, 64,  [96, 256, 512]);
  uni(G_MAC, 'Mac Studio · M5 Ultra 80-core GPU',          1200, 80,  [96, 256, 512], 512);
  uni(G_MAC, 'MacBook Pro 14" · M5',                       153,  10,  [16, 24, 32]);
  uni(G_MAC, 'MacBook Pro 14"/16" · M5 Pro 16-core GPU',   307,  16,  [16, 24, 32]);
  uni(G_MAC, 'MacBook Pro 14"/16" · M5 Pro 20-core GPU',   307,  20,  [24, 36, 48, 64, 128]);
  uni(G_MAC, 'MacBook Pro 14"/16" · M5 Max 32-core GPU',   460,  32,  [36, 48, 64, 128]);
  uni(G_MAC, 'MacBook Pro 14"/16" · M5 Max 40-core GPU',   614,  40,  [36, 48, 64, 128]);
  uni(G_MAC, 'MacBook Air 13"/15" · M5',                   153,  10,  [16, 24, 32]);
  uni(G_MAC, 'iMac · M4',                                  120,  4.6, [16, 24, 32]);

  // --- Apple: прошлые чипы (Mac mini, Studio, MacBook 2020–2025, б/у рынок)
  const G_OLD = 'Apple · прошлые чипы';
  uni(G_OLD, 'M1',        68,  2.6,  [8, 16]);
  uni(G_OLD, 'M1 Pro',    200, 5.2,  [16, 32]);
  uni(G_OLD, 'M1 Max',    400, 10.4, [32, 64]);
  uni(G_OLD, 'M1 Ultra',  800, 21,   [64, 128]);
  uni(G_OLD, 'M2',        100, 3.6,  [8, 16, 24]);
  uni(G_OLD, 'M2 Pro',    200, 6.8,  [16, 32]);
  uni(G_OLD, 'M2 Max',    400, 13.6, [32, 64, 96]);
  uni(G_OLD, 'M2 Ultra',  800, 27,   [64, 128, 192]);
  uni(G_OLD, 'M3',        100, 4.1,  [8, 16, 24]);
  uni(G_OLD, 'M3 Pro',    150, 7,    [18, 36]);
  uni(G_OLD, 'M3 Max 30-core GPU', 300, 10.6, [36, 96]);
  uni(G_OLD, 'M3 Max 40-core GPU', 400, 14,   [48, 64, 128]);
  uni(G_OLD, 'M3 Ultra',  800, 28,   [96, 256, 512]);
  uni(G_OLD, 'M4',        120, 4.6,  [16, 24, 32]);
  uni(G_OLD, 'M4 Pro',    273, 9.2,  [24, 48, 64]);
  uni(G_OLD, 'M4 Max 32-core GPU', 410, 14.5, [36]);
  uni(G_OLD, 'M4 Max 40-core GPU', 546, 18,   [48, 64, 128]);
  uni(G_OLD, 'M5 (первые MacBook Pro 14", 2025)', 153, 10, [16, 24, 32]);

  // --- Коробки на NVIDIA GB10: одна платформа, 64 (с 23.10.2026) или 128 ГБ LPDDR5X 273 ГБ/с, 1 PFLOP FP4 (≈60 TFLOPS FP16 практичных)
  const G_GB10 = 'Коробки NVIDIA GB10';
  for (const n of ['NVIDIA DGX Spark', 'ASUS Ascent GX10', 'Dell Pro Max with GB10', 'HP ZGX Nano AI Station G1n', 'Lenovo ThinkStation PGX',
                   'MSI EdgeXpert MS-C931', 'Acer Veriton GN100', 'Gigabyte AI TOP Atom'])
    uni(G_GB10, n, 273, 60, [64, 128], 128);
  uni(G_GB10, '2× DGX Spark в связке (ConnectX-7)', 273, 60, [256], 256);
  // DGX OS (Linux): CUDA видит почти всю общую память, системе остаётся ~10 %
  for (const h of HARDWARE) if (h.group === G_GB10) h.gpuShare = 0.9;

  // --- Коробки на AMD Strix Halo (Ryzen AI Max+ 395: 40 CU, 256 ГБ/с; GPU отдают до 96 из 128 ГБ)
  const G_SH = 'Коробки AMD Strix Halo';
  const sh = (n, mems, tf = 15, bw = 256) => uni(G_SH, n, bw, tf, mems, mems[mems.length - 1]);
  sh('Framework Desktop (AI Max+ 395)', [64, 128]);
  sh('Framework Desktop (AI Max 385, 32 CU)', [32], 12);
  sh('Framework Desktop (AI Max+ PRO 495, 192 ГБ)', [192], 15, 273);   // Gorgon Halo: LPDDR5X-8533, GPU до 160 ГБ
  HARDWARE[HARDWARE.length - 1].gpuShare = 160 / 192;
  sh('GMKtec EVO-X2', [64, 96, 128]);
  sh('Beelink GTR9 Pro', [64, 128]);
  sh('Minisforum MS-S1 MAX', [128]);
  sh('HP Z2 Mini G1a', [32, 64, 128]);
  sh('Bosgame M5', [64, 96, 128]);
  sh('Corsair AI Workstation 300', [32, 64, 128]);
  sh('Sixunited AXB35 / AXB88', [64, 128]);
  sh('Acemagic Tank M1A Pro Plus', [128]);
  sh('Thunderobot Station', [128]);
  sh('Nimo AI PC (AI Max+ 395)', [128]);
  sh('ASUS ROG Flow Z13 / HP ZBook Ultra G1a (ноутбуки)', [32, 64, 128]);

  // --- NVIDIA Jetson (единая память, LPDDR5)
  const G_JET = 'NVIDIA Jetson';
  uni(G_JET, 'Jetson Orin Nano Super 8 ГБ', 102, 8,   [8]);
  uni(G_JET, 'Jetson Orin NX 16 ГБ',        102, 12,  [16]);
  uni(G_JET, 'Jetson AGX Orin 32 ГБ',       205, 35,  [32]);
  uni(G_JET, 'Jetson AGX Orin 64 ГБ',       205, 35,  [64]);
  uni(G_JET, 'Jetson AGX Thor 128 ГБ',      273, 130, [128]);
  for (const h of HARDWARE) if (h.group === G_JET) h.gpuShare = 0.9;   // JetPack (Linux), как у GB10

  // --- Мини-ПК и ноутбуки на других чипах (GPU встроенный, память общая)
  const G_MINI = 'Мини-ПК на других чипах';
  uni(G_MINI, 'Intel Core Ultra X9 388H Panther Lake (GMKtec EVO-T2 и др.)', 136, 15, [32, 64, 96, 128]);
  uni(G_MINI, 'Intel Core Ultra 9 285H Arrow Lake (ASUS NUC 15 Pro и др.)',  102, 8,  [32, 64, 96]);
  uni(G_MINI, 'AMD Ryzen AI 9 HX 370 (Beelink SER9, GEEKOM A9 Max, EVO-X1)',   90, 6,  [32, 64, 96]);
  uni(G_MINI, 'Qualcomm Snapdragon X2 Elite Extreme',                          228, 6,  [48, 64, 128]);
  uni(G_MINI, 'Qualcomm Snapdragon X2 Elite',                                  152, 5,  [32, 48, 64, 128]);
  uni(G_MINI, 'Qualcomm Snapdragon X Elite',                                   135, 4.6, [16, 32, 64]);

  // --- Дискретные карты NVIDIA
  add('NVIDIA', 'RTX 3060 12 ГБ',       12, 360,  51);
  add('NVIDIA', 'RTX 3080 10 ГБ',       10, 760,  119);
  add('NVIDIA', 'RTX 3090 24 ГБ',       24, 936,  142);
  add('NVIDIA', 'RTX 4060 Ti 16 ГБ',    16, 288,  88);
  add('NVIDIA', 'RTX 4070 12 ГБ',       12, 504,  116);
  add('NVIDIA', 'RTX 4070 Ti Super 16 ГБ', 16, 672, 158);
  add('NVIDIA', 'RTX 4080 16 ГБ',       16, 717,  195);
  add('NVIDIA', 'RTX 4090 24 ГБ',       24, 1008, 165);
  add('NVIDIA', 'RTX 5060 Ti 16 ГБ',    16, 448,  95);
  add('NVIDIA', 'RTX 5070 12 ГБ',       12, 672,  123);
  add('NVIDIA', 'RTX 5070 Ti 16 ГБ',    16, 896,  176);
  add('NVIDIA', 'RTX 5080 16 ГБ',       16, 960,  225);
  add('NVIDIA', 'RTX 5090 32 ГБ',       32, 1792, 210);
  add('NVIDIA', 'RTX A6000 48 ГБ',      48, 768,  155);
  add('NVIDIA', 'RTX 6000 Ada 48 ГБ',   48, 960,  182);
  add('NVIDIA', 'RTX PRO 6000 Blackwell 96 ГБ', 96, 1792, 250);
  add('NVIDIA (сервер)', 'A100 80 ГБ',  80, 2039, 312, { sysMemGB: 256, sysBandwidthGBs: 200 });
  add('NVIDIA (сервер)', 'H100 SXM 80 ГБ', 80, 3350, 990, { sysMemGB: 512, sysBandwidthGBs: 300 });
  add('NVIDIA (сервер)', 'H200 141 ГБ', 141, 4800, 990, { sysMemGB: 512, sysBandwidthGBs: 300 });
  add('NVIDIA (сервер)', 'DGX Station GB300 (252 ГБ HBM3e + 496 ГБ LPDDR5X)', 252, 7100, 1100, { sysMemGB: 496, sysBandwidthGBs: 396, sysTflops: 4 });
  add('AMD', 'RX 7900 XTX 24 ГБ',       24, 960,  123);
  add('AMD', 'RX 9070 XT 16 ГБ',        16, 640,  97);
  add('AMD', 'Radeon AI PRO R9700 32 ГБ', 32, 640, 96);
  add('Intel', 'Arc B580 12 ГБ',        12, 456,  58);
  add('Intel', 'Arc Pro B70 32 ГБ',     32, 608,  91);
  const cpu = (name, mem, bw, tf, mems) => add('Только процессор', name, mem, bw, tf, { unified: true, cpuOnly: true, memOptions: mems });
  cpu('Ноутбук, DDR4-3200 двухканал',      16,  51,  0.8, [8, 16, 32, 64]);
  cpu('ПК, DDR5-5600 двухканал',           32,  90,  1.5, [16, 32, 64, 96, 128, 192]);
  cpu('ПК, DDR5-6400 двухканал',           32,  102, 1.5, [16, 32, 64, 96, 128, 192]);
  cpu('Рабочая станция, DDR5 8-канал',     256, 300, 4,   [128, 256, 512, 1024]);
  cpu('Сервер, DDR5 12-канал (EPYC)',      512, 460, 6,   [256, 512, 1024, 2048]);

  // Оболочки (движки вывода). modes — куда можно положить то, что не влезло в память ускорителя:
  // layers — слои в ОЗУ; experts — эксперты MoE в ОЗУ, внимание/общие веса/KV на GPU; ssd — эксперты подкачиваются с SSD.
  // ssd: true — n-gram/Engram-таблицы (ssdParams) читаются с диска и не занимают память. kv — какие кэши поддерживает.
  // Источники: README и документация движков, проверено 01.11.2026.
  const ALL = ['apple', 'nvidia', 'amd', 'intel', 'qualcomm', 'cpu'];
  const ENGINES = {
    'llama.cpp': { label: 'llama.cpp (и Ollama, LM Studio)', vendors: ALL, modes: ['layers', 'experts'], kv: ['FP16', 'Q8_0', 'Q4_0'],
      fmts: ['gguf', 'any'], mtp: hw => hw.vendor !== 'apple' && !hw.cpuOnly, syncMs: 0.3,
      note: 'Универсальная, любая платформа. Не влезло — слои (-ngl) или эксперты MoE (--n-cpu-moe) уходят в ОЗУ. MTP-головы (--spec-type draft-mtp) с мая 2026; на Mac пока без выигрыша. Ollama и LM Studio работают на ней же.' },
    mlx: { label: 'MLX / MTPLX (только Mac)', vendors: ['apple'], modes: [], ssd: true, kv: ['FP16', 'Q8_0', 'Q4_0'],
      fmts: ['mlx', 'any'], mtp: () => true,
      note: 'Только Mac. MTPLX использует MTP-голову модели: генерация ×1,6–2,7. N-gram-таблицы Qwen3.8-Flash-Next читаются с SSD, в памяти их нет.' },
    vllm: { label: 'vLLM / SGLang', vendors: ['nvidia', 'amd', 'intel'], modes: [], usable: 0.9, kv: ['FP16', 'Q8_0', 'Q4_0', 'TQ3'],
      fmts: ['any', 'fp4', 'gptq'], mtp: () => true,
      note: 'Серверная: всё в видеопамяти (занимает 90 %), KV в FP8, NVFP4 или TurboQuant 3 бит, MTP и EAGLE. Выигрывает при многих запросах сразу; одному пользователю llama.cpp не уступает.' },
    ktrans: { label: 'KTransformers / ik_llama.cpp (CPU + GPU)', vendors: ['nvidia', 'amd'], discreteOnly: true, modes: ['experts'], kv: ['FP16', 'Q8_0'], syncMs: 0.15,
      fmts: ['gguf', 'any'], mtp: () => true,
      note: 'Для больших MoE на одной видеокарте: экспертов в ОЗУ считает процессор (AMX, AVX-512), внимание и KV — на видеокарте. MTP у GLM и Qwen 3.5+.' },
    strata: { label: 'Strata (Flash-Next, Qwen3.6, DeepSeek V4.1)', vendors: ['nvidia', 'amd', 'intel', 'apple'], strataOnly: true, minVramGB: 12,
      modes: ['experts', 'ssd'], ssd: true, kv: ['FP16'], fmts: ['gguf', 'fp4', 'any'], mtp: () => true, syncMs: 0.07, ubatch: 4096,
      note: 'Узкая: только Qwen3.8-Flash-Next, Qwen3.6 и DeepSeek V4.1 Flash. На видеокарте внимание, общие эксперты, KV и кэш горячих экспертов; остальные эксперты в ОЗУ, n-gram/Engram — на SSD; MTP ×1,6–1,8. Нужна карта от 12 ГБ.' },
    stream: { label: 'Подкачка с SSD (flash-moe, AirLLM)', vendors: ALL, moeOnly: true, modes: ['ssd'], ssd: true, kv: ['FP16', 'Q8_0', 'Q4_0'],
      fmts: ['mlx', 'gguf', 'any'], mtp: () => false,
      note: 'Эксперты MoE читаются с SSD по мере надобности, в памяти только горячие. Запускает модели больше памяти, но это единицы ток/с: для проб, не для работы.' },
  };
  const NON_EXPERT_SHARE = 0.4;   // доля активных весов MoE вне экспертов: внимание, роутер, общие эксперты (оценка)
  const MTP_GAIN = 1.6;           // MTP-голова как черновик: замеры 1,6–1,8 (Strata), 1,6–2,7 (MTPLX), 1,4–1,7 (llama.cpp на CUDA)
  const PCIE_GBs = 25;            // PCIe 4.0 x16 на практике: по ней при чтении промпта гоняются эксперты из ОЗУ
  const UBATCH = 2048;            // токенов промпта на один проход экспертов
  const SSD_HIT_POWER = 0.2;
  // Задержка на слой при генерации (запуск ядер, синхронизация), мс. Калибровка: gpt-oss-120b на M5 Max — 76 ток/с
  // против 75–79 в замере (без неё 129); Qwen3.5-35B-A3B на RTX 5090 — 194 в замере. Плюс syncMs оболочки на слой,
  // когда часть работы на процессоре: llama.cpp --n-cpu-moe на RTX 3060 — 41 ток/с в замере.
  const LAYER_MS = { apple: 0.19, cpu: 0.02 }, LAYER_MS_GPU = 0.09;      // попадание в кэш горячих экспертов = (доля кэша)^0,2: калибровка по flash-moe (15 % кэша → ~70 %)

  const vendorOfHw = hw => hw.vendor ?? (hw.cpuOnly ? 'cpu' : hw.unified ? 'apple' : 'nvidia');
  const isMoe = model => model.active < model.params * 0.5;
  function engineOk(key, model, hw) {
    const e = ENGINES[key];
    if (!e || !e.vendors.includes(vendorOfHw(hw))) return false;
    if (e.strataOnly && !model.strata) return false;
    if (e.moeOnly && !isMoe(model)) return false;
    if (e.discreteOnly && hw.unified) return false;
    if (e.minVramGB && !hw.unified && hw.memGB < e.minVramGB) return false;
    return true;
  }

  function weightsBytes(params, bpw) { return params * bpw / 8; }

  function kvBytesPerToken(model, kvBits, ctx) {
    const factor = model.kvFactor ?? 2;
    const layers = model.kvLayers ?? model.layers;   // у гибридов KV-кэш растёт только в слоях полного внимания
    const full = factor * layers * model.kvHeads * model.headDim * kvBits / 8 * (model.kvScale ?? 1);
    if (model.window && model.localRatio && ctx > model.window) {
      return full * ((1 - model.localRatio) + model.localRatio * model.window / ctx);
    }
    return full;
  }

  // Накладные: сама программа + служебные тензоры (~4 % весов) + буфер вычислений (0,25 ГБ + растёт с контекстом; считаем с flash attention)
  function overheadBytes(weights, model, ctx) {
    return RUNTIME_BYTES + 0.04 * weights + 0.25 * GB + ctx * model.hidden * 2;
  }

  // ssd — n-gram/Engram-таблицы, которые оболочка держит на диске (в память они не идут)
  function memoryNeeded({ model, bpw, ctx, kvBits, engine }) {
    const onSsd = engine && ENGINES[engine]?.ssd ? (model.ssdParams ?? 0) : 0;
    const weights = weightsBytes(model.params - onSsd, bpw);
    const kv = kvBytesPerToken(model, kvBits, ctx) * ctx;
    const overhead = overheadBytes(weights, model, ctx);
    return { weights, kv, overhead, ssd: weightsBytes(onSsd, bpw), total: weights + kv + overhead };
  }

  // На каждый новый токен читаются активные веса и весь KV-кэш
  function bytesPerGeneratedToken({ model, bpw, ctx, kvBits }) {
    return weightsBytes(model.active, bpw) + kvBytesPerToken(model, kvBits, ctx) * ctx;
  }

  // Внимание при чтении промпта длиной P: в среднем 2 × скрытая × P FLOP на токен в слое полного внимания,
  // в слое со скользящим окном W — не больше 4 × скрытая × W. На коротких промптах почти ноль, на 100K+ — главное.
  function attnFlopsPerToken(model, P) {
    if (!P) return 0;
    const layers = model.kvLayers ?? model.layers, d = model.hidden;
    const local = model.window && model.localRatio ? model.localRatio : 0;
    const W = model.window ?? P;
    const localAvg = P <= W ? 2 * d * P : 4 * d * W * (1 - W / (2 * P));
    return layers * ((1 - local) * 2 * d * P + local * localAvg);
  }

  function genTokPerSec(bandwidthGBs, bytesPerToken) { return bandwidthGBs * 1e9 * BW_EFF / bytesPerToken; }
  // Постоянная часть времени на токен: слои × задержка; hybrid — часть работы на процессоре (выгрузка)
  function stepLatencySec(model, hw, hybrid = false, engine = null) {
    const per = LAYER_MS[vendorOfHw(hw)] ?? LAYER_MS_GPU;
    const sync = hybrid ? (engine ? ENGINES[engine].syncMs ?? 0.3 : 0.3) : 0;
    return model.layers * (per + sync) / 1000;
  }
  function prefillTokPerSec(tflops, model, promptTokens = 0) {
    return tflops * 1e12 * COMPUTE_EFF / (2 * model.active + attnFlopsPerToken(model, promptTokens));
  }
  // latencySec — постоянная часть времени на токен: если цель быстрее неё, никакая память не поможет (Infinity)
  function requiredBandwidthGBs(bytesPerToken, targetTps, latencySec = 0) {
    const left = 1 / targetTps - latencySec;
    return left > 0 ? bytesPerToken / left / BW_EFF / 1e9 : Infinity;
  }
  function requiredTflops(model, promptTokens, seconds) {
    return (2 * model.active + attnFlopsPerToken(model, promptTokens)) * (promptTokens / seconds) / COMPUTE_EFF / 1e12;
  }

  // macOS по умолчанию отдаёт GPU 2/3 памяти при ≤36 ГБ и 3/4 выше (можно поднять через sysctl, но по умолчанию так).
  // У коробок на Linux (GB10, Jetson, Gorgon Halo) доля своя — gpuShare.
  function usableMemoryBytes(hw) {
    const total = hw.memGB * GB;
    if (hw.cpuOnly) return total * OS_RESERVE_SYS;
    if (hw.unified) return total * (hw.gpuShare ?? (hw.memGB <= 36 ? 2 / 3 : 3 / 4));
    return total;
  }

  // cfg.engine — ключ ENGINES (без него — как llama.cpp без MTP: только выгрузка слоёв); cfg.promptTokens — для цены внимания
  function fit(cfg, hw) {
    const e = cfg.engine ? ENGINES[cfg.engine] : null;
    const modes = e ? e.modes : ['layers'];
    const { model, bpw } = cfg;
    const mem = memoryNeeded(cfg);
    const usable = usableMemoryBytes(hw) * (e?.usable ?? 1);
    const bytes = bytesPerGeneratedToken(cfg);
    const mtpGain = e && model.mtp && e.mtp({ ...hw, vendor: vendorOfHw(hw) }) ? MTP_GAIN : 1;
    const gpuBw = hw.bandwidthGBs * 1e9 * BW_EFF;
    const gpuPre = prefillTokPerSec(hw.tflops, model, cfg.promptTokens);
    const base = { need: mem.total, usable, mem, bytesPerToken: bytes, engine: cfg.engine, mtpGain };
    const lat = stepLatencySec(model, hw), latHybrid = stepLatencySec(model, hw, true, cfg.engine);
    if (mem.total <= usable) {
      return { ...base, status: 'fits', gpuFraction: 1, genTps: mtpGain / (bytes / gpuBw + lat), prefillTps: gpuPre, latency: lat };
    }
    const sysUsable = hw.unified ? 0 : (hw.sysMemGB ?? 0) * GB * OS_RESERVE_SYS;
    const sysBw = (hw.sysBandwidthGBs ?? 0) * 1e9 * BW_EFF;
    const options = [];
    if (modes.includes('layers') && !hw.unified && mem.total <= usable + sysUsable) {
      const f = usable / mem.total;
      const tGen = f * bytes / gpuBw + (1 - f) * bytes / sysBw + latHybrid;
      const tPre = f / gpuPre + (1 - f) / prefillTokPerSec(hw.sysTflops ?? 1, model, cfg.promptTokens);
      options.push({ status: 'offload', gpuFraction: f, genTps: mtpGain / tGen, prefillTps: 1 / tPre, latency: latHybrid });
    }
    // MoE: на GPU внимание, общие веса и KV; эксперты — в ОЗУ (experts) или на SSD с кэшем горячих в памяти (ssd)
    if (isMoe(model) && (modes.includes('experts') || modes.includes('ssd'))) {
      const act = weightsBytes(model.active, bpw), shared = act * NON_EXPERT_SHARE, expAct = act - shared;
      const experts = mem.weights - shared;
      const gpuNeed = shared + mem.kv + mem.overhead;
      if (gpuNeed <= usable) {
        const tShared = (shared + mem.kv) / gpuBw + latHybrid;
        const ub = e?.ubatch ?? UBATCH;
        const spare = usable - gpuNeed;
        if (modes.includes('experts') && !hw.unified) {
          const onGpu = Math.min(experts, spare), ramNeed = experts - onGpu, c = onGpu / experts;
          if (ramNeed <= sysUsable) {
            const t = tShared + expAct * (c / gpuBw + (1 - c) / sysBw);
            options.push({ status: 'experts', gpuFraction: (gpuNeed + onGpu) / mem.total, gpuNeed: gpuNeed + onGpu, ramNeed,
              genTps: mtpGain / t, prefillTps: 1 / (1 / gpuPre + ramNeed / (PCIE_GBs * 1e9) / ub), latency: latHybrid });
          }
        }
        if (modes.includes('ssd')) {
          const cache = spare + sysUsable, ramBw = hw.unified ? gpuBw : sysBw;
          const tBase = hw.unified ? (shared + mem.kv) / gpuBw + lat : tShared;
          const hit = Math.min(0.9, Math.pow(Math.min(1, cache / experts), SSD_HIT_POWER));
          const ssdBw = (hw.ssdGBs ?? 6) * 1e9;
          const t = tBase + expAct * (hit / ramBw + (1 - hit) / ssdBw);
          options.push({ status: 'ssd', gpuFraction: gpuNeed / mem.total, gpuNeed, ramNeed: Math.min(cache, experts), ssdHit: hit,
            genTps: mtpGain / t, prefillTps: 1 / (1 / gpuPre + experts * (1 - hit) / ssdBw / ub), latency: tBase - (shared + mem.kv) / gpuBw });
        }
      }
    }
    if (!options.length) return { ...base, status: 'nofit', gpuFraction: 0, genTps: 0, prefillTps: 0 };
    options.sort((a, b) => b.genTps - a.genTps);
    return { ...base, ...options[0] };
  }

  const RUNS = ['fits', 'experts', 'offload', 'ssd'];
  // Наибольший контекст, при котором модель остаётся в одном из режимов statuses (по умолчанию — целиком в памяти ускорителя)
  function maxContext(cfg, hw, statuses = ['fits']) {
    let lo = 0, hi = 1 << 22;
    if (!statuses.includes(fit({ ...cfg, ctx: lo }, hw).status)) return 0;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (statuses.includes(fit({ ...cfg, ctx: mid }, hw).status)) lo = mid; else hi = mid;
    }
    return lo;
  }

  // Эксперты в ОЗУ у MoE — рабочий режим (так задуманы Strata и KTransformers), поэтому может быть «удобно»
  function verdict(f, goal) {
    if (f.status === 'nofit') return 'bad';
    const promptSec = goal.promptTokens / f.prefillTps;
    const genOk = f.genTps >= goal.genTps, preOk = promptSec <= goal.promptSeconds;
    if ((f.status === 'fits' || f.status === 'experts') && genOk && preOk) return 'good';
    if (f.genTps >= goal.genTps / 2 && promptSec <= goal.promptSeconds * 3) return 'ok';
    return 'bad';
  }

  // KV-кэш, который оболочка умеет: если выбранного нет — ближайший поддерживаемый не меньше (иначе FP16)
  function kvFor(engine, kvBits) {
    const ok = ENGINES[engine].kv.map(k => KV_QUANTS[k]).sort((a, b) => a - b);
    return ok.find(b => b >= kvBits) ?? ok[ok.length - 1];
  }

  // Все оболочки, которые запустятся на этом железе с этой моделью, от лучшей к худшей
  function rankEngines(cfg, hw) {
    const score = f => (f.status === 'fits' || f.status === 'experts' ? 2 : f.status === 'nofit' ? 0 : 1) * 1e6 + f.genTps;
    return Object.keys(ENGINES).filter(k => engineOk(k, cfg.model, hw))
      .map(k => ({ engine: k, fit: fit({ ...cfg, engine: k, kvBits: kvFor(k, cfg.kvBits) }, hw) }))
      .sort((a, b) => score(b.fit) - score(a.fit));
  }
  function bestEngine(cfg, hw) { return rankEngines(cfg, hw)[0] ?? null; }

  const quantFits = (k, engine) => (engine ? ENGINES[engine].fmts : ['gguf', 'any']).includes(QUANTS[k].fmt);

  // Диагностика: во что упёрлись и что с этим делать. Возвращает числа, текст собирает интерфейс.
  function diagnose(cfg, hw, goal) {
    const f = fit(cfg, hw);
    const promptSec = f.prefillTps ? goal.promptTokens / f.prefillTps : Infinity;
    const kvShare = f.mem.kv / (f.mem.weights + f.mem.kv);
    const moe = isMoe(cfg.model);
    const ok = s => s === 'fits';
    const byQuality = Object.keys(QUANTS).filter(k => quantFits(k, cfg.engine)).sort((a, b) => QUANTS[b].bpw - QUANTS[a].bpw);
    let bestQuantFits = null, bestQuantMeetsTarget = null;
    for (const k of byQuality) {
      const ff = fit({ ...cfg, bpw: QUANTS[k].bpw }, hw);
      if (!ok(ff.status)) continue;
      if (!bestQuantFits) bestQuantFits = k;
      if (!bestQuantMeetsTarget && ff.genTps >= goal.genTps) bestQuantMeetsTarget = k;
    }
    const kvOk = k => !cfg.engine || ENGINES[cfg.engine].kv.includes(k);
    const kvQ8Fits = kvOk('Q8_0') && cfg.kvBits > KV_QUANTS.Q8_0 && ok(fit({ ...cfg, kvBits: KV_QUANTS.Q8_0 }, hw).status);
    const kvQ4Fits = kvOk('Q4_0') && cfg.kvBits > KV_QUANTS.Q4_0 && ok(fit({ ...cfg, kvBits: KV_QUANTS.Q4_0 }, hw).status);
    // macOS по умолчанию режет память GPU; sysctl iogpu.wired_limit_mb поднимает её примерно до 90 %
    const macWiredLimitHelps = !!(hw.unified && !hw.cpuOnly && vendorOfHw(hw) === 'apple' && f.status !== 'fits' && f.need <= hw.memGB * GB * 0.9);
    // MoE: эксперты в ОЗУ, внимание и KV на GPU — читаются только активные веса через шину ОЗУ
    // (для nofit — оценка «если бы ОЗУ хватило», чтобы совет добавить память был с числом)
    let expertOffloadTps = null;
    if (moe && !hw.unified && f.status !== 'fits') {
      const kvBytes = kvBytesPerToken(cfg.model, cfg.kvBits, cfg.ctx) * cfg.ctx;
      const t = weightsBytes(cfg.model.active, cfg.bpw) / (hw.sysBandwidthGBs * 1e9 * BW_EFF) + kvBytes / (hw.bandwidthGBs * 1e9 * BW_EFF);
      expertOffloadTps = 1 / t;
    }
    let bottleneck;
    if (f.status === 'nofit') bottleneck = 'memory';
    else if (f.status !== 'fits' && f.status !== 'experts') bottleneck = 'offload';
    else if (f.genTps < goal.genTps) bottleneck = 'bandwidth';
    else if (promptSec > goal.promptSeconds) bottleneck = 'compute';
    else bottleneck = 'none';
    // Другие оболочки, которые на этом железе дадут больше
    const engines = rankEngines(cfg, hw);
    const better = engines.filter(r => r.engine !== cfg.engine && r.fit.status !== 'nofit'
      && (f.status === 'nofit' || (RUNS.indexOf(r.fit.status) < RUNS.indexOf(f.status)) || r.fit.genTps > f.genTps * 1.15));
    return { bottleneck, fit: f, promptSec, kvShare, moe, bestQuantFits, bestQuantMeetsTarget, kvQ8Fits, kvQ4Fits,
      macWiredLimitHelps, expertOffloadTps, engines, betterEngines: better,
      requiredBandwidthGBs: requiredBandwidthGBs(f.bytesPerToken, goal.genTps / (f.mtpGain ?? 1), stepLatencySec(cfg.model, hw)),
      requiredTflops: requiredTflops(cfg.model, goal.promptTokens, goal.promptSeconds) };
  }

  // Число слоёв для своей модели: интерполяция по известным архитектурам в логарифме числа параметров
  const LAYER_ANCHORS = [[1e9, 16], [3e9, 28], [8e9, 32], [14e9, 40], [32e9, 64], [70e9, 80], [405e9, 126]];
  function estimateLayers(params) {
    if (params <= LAYER_ANCHORS[0][0]) return LAYER_ANCHORS[0][1];
    for (let i = 1; i < LAYER_ANCHORS.length; i++) {
      const [p0, l0] = LAYER_ANCHORS[i - 1], [p1, l1] = LAYER_ANCHORS[i];
      if (params <= p1) {
        const t = (Math.log(params) - Math.log(p0)) / (Math.log(p1) - Math.log(p0));
        return 2 * Math.round((l0 + t * (l1 - l0)) / 2);
      }
    }
    return LAYER_ANCHORS[LAYER_ANCHORS.length - 1][1];
  }

  return { GB, BW_EFF, COMPUTE_EFF, QUANTS, KV_QUANTS, KV_LABELS, MODELS, HARDWARE, ENGINES, NON_EXPERT_SHARE, MTP_GAIN,
    weightsBytes, kvBytesPerToken, memoryNeeded, bytesPerGeneratedToken, genTokPerSec, prefillTokPerSec, attnFlopsPerToken, stepLatencySec,
    requiredBandwidthGBs, requiredTflops, usableMemoryBytes, fit, maxContext, verdict, diagnose, estimateLayers,
    engineOk, kvFor, rankEngines, bestEngine };
})();
