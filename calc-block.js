// Формулы и справочники калькулятора. Всё в одном объекте LLMCalc, без DOM.
var LLMCalc = (() => {
  const GB = 1024 ** 3;
  const BW_EFF = 0.7;       // реальная доля пропускной способности, которую достигают llama.cpp / MLX / vLLM при генерации
  const COMPUTE_EFF = 0.5;  // реальная доля пиковых TFLOPS при обработке промпта
  const RUNTIME_BYTES = 0.5 * GB;   // сама программа, драйвер, CUDA-контекст
  const OS_RESERVE_SYS = 0.85;      // сколько системной ОЗУ можно занять под выгрузку

  // Реальные биты на вес у GGUF-квантов (с учётом масштабов и смешанных типов слоёв)
  const QUANTS = {
    FP16:    { bpw: 16,   label: 'FP16 / BF16', note: 'без потерь, вдвое больше Q8' },
    Q8_0:    { bpw: 8.5,  label: 'Q8_0',        note: 'почти без потерь' },
    Q6_K:    { bpw: 6.56, label: 'Q6_K',        note: 'потери незаметны' },
    Q5_K_M:  { bpw: 5.69, label: 'Q5_K_M',      note: 'лёгкие потери' },
    Q4_K_M:  { bpw: 4.85, label: 'Q4_K_M',      note: 'стандарт: баланс размера и качества' },
    MXFP4:   { bpw: 4.25, label: 'MXFP4',       note: 'родной формат GPT-OSS' },
    Q4_0:    { bpw: 4.55, label: 'Q4_0',        note: 'быстрый на Apple, чуть хуже K_M' },
    Q3_K_M:  { bpw: 3.91, label: 'Q3_K_M',      note: 'заметные потери' },
    Q2_K:    { bpw: 3.35, label: 'Q2_K',        note: 'сильные потери' },
    IQ2_XXS: { bpw: 2.06, label: 'IQ2_XXS',     note: 'крайний случай, качество падает' },
  };
  const KV_QUANTS = { FP16: 16, Q8_0: 8, Q4_0: 4 };

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
  m('Qwen 3.5–3.8', 'Qwen3.8 2.4T-A95B (MoE)',      2400, 95,   92, 4, 256, 8192, 262144, { kvLayers: 23 });
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
  // --- DeepSeek
  m('DeepSeek', 'DeepSeek-V2-Lite 16B-A2.4B (MoE, MLA)', 15.7, 2.4, 27, 1, 576, 2048, 32768, MLA);
  m('DeepSeek', 'DeepSeek-V2 / Coder-V2 236B-A21B (MoE, MLA)', 236, 21, 60, 1, 576, 5120, 131072, MLA);
  m('DeepSeek', 'DeepSeek V3 / V3.1 / V3.2 / R1 671B-A37B (MoE, MLA)', 671, 37, 61, 1, 576, 7168, 131072, MLA);
  // V4 (2026): сжатое разрежённое внимание, по карточке модели KV ≈ 10 % от V3.2
  m('DeepSeek', 'DeepSeek V4 Flash 284B-A13B (MoE)', 284, 13, 43, 1, 576, 4096, 1048576, { ...MLA, kvScale: 0.1, note: 'KV ≈ 10 % от V3 по карточке' });
  m('DeepSeek', 'DeepSeek V4 Pro 1.6T-A49B (MoE)',   1600, 49, 61, 1, 576, 7168, 1048576, { ...MLA, kvScale: 0.1, note: 'KV ≈ 10 % от V3 по карточке' });
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
  m('GLM', 'GLM-5 / 5.1 / 5.2 744B-A40B (MoE, MLA)', 744, 40, 78, 1, 576, 6144, 1048576, MLA);
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
  m('Китайские MoE', 'MiMo-V2-Flash 309B-A15B (MoE)', 309, 15, 48, 4, 192, 4096, 262144, SW(128, 39/48));
  m('Китайские MoE', 'MiMo-V2.5 310B-A15B (MoE)', 310, 15, 48, 4, 192, 4096, 1048576, SW(128, 39/48));
  m('Китайские MoE', 'MiMo-V2.5-Pro 1T-A42B (MoE)', 1020, 42, 70, 8, 192, 6144, 1048576, SW(128, 6/7));
  m('Китайские MoE', 'Ring-2.6-1T 1T-A63B (MoE, MLA)', 1000, 63, 80, 1, 576, 8192, 131072, MLA);
  m('Китайские MoE', 'Hunyuan-A13B 80B (MoE)', 80, 13, 32, 8, 128, 4096, 262144);
  m('Китайские MoE', 'ERNIE 4.5 21B-A3B (MoE)', 21, 3, 28, 4, 128, 2560, 131072);
  m('Китайские MoE', 'ERNIE 4.5 300B-A47B (MoE)', 300, 47, 54, 8, 128, 8192, 131072);
  m('Китайские MoE', 'Seed-OSS 36B',      36,   36,   64, 8,  128, 5120,  524288);
  // --- NVIDIA Nemotron 3 (гибрид Mamba-2 + MoE, внимание лишь в части слоёв)
  m('NVIDIA', 'Nemotron 3 Nano 30B-A3B',  31.6, 3.5,  52, 2,  128, 2688,  262144, { kvLayers: 6 });
  m('NVIDIA', 'Nemotron 3 Super 120B-A12B', 120, 12,  88, 2,  128, 4096,  262144, { kvLayers: 9 });
  m('NVIDIA', 'Nemotron 3 Ultra 550B-A55B', 550, 55,  128, 2, 128, 8192,  262144, { kvLayers: 16 });
  // --- Poolside
  m('Poolside', 'Laguna XS 2.1 33B-A3B (MoE)', 33, 3, 40, 8, 128, 2048, 262144, SW(512, 3/4));
  m('Poolside', 'Laguna S 2.1 118B-A8B (MoE)', 118, 8, 48, 8, 128, 3072, 1048576, SW(512, 3/4));
  // --- Cohere
  m('Cohere', 'Command R 35B',            35,   35,   40, 8,  128, 8192,  131072);
  m('Cohere', 'Command R+ 104B',          104,  104,  64, 8,  128, 12288, 131072);
  m('Cohere', 'Command A 111B',           111,  111,  64, 8,  128, 12288, 262144, SW(4096, 3/4));
  // --- Прочие открытые
  m('Прочие', 'TinyLlama 1.1B',           1.1,  1.1,  22, 4,  64,  2048,  2048);
  m('Прочие', 'SmolLM3 3B',               3.1,  3.1,  36, 4,  128, 2048,  65536);
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

  // Железо. memGB — память ускорителя (или вся ОЗУ при unified). tflops — практичные FP16 без разрежённости
  // (у NVIDIA это половина маркетинговых tensor-цифр, у Apple M1–M4 — пиковые FP32; у M5/M6 с нейроускорителями в GPU
  // взято ×2 к M4-аналогу, консервативно к заявленным ×3–4). Для дискретных карт sysMemGB/sysBandwidthGBs — ОЗУ под выгрузку.
  // Коробки и Mac проверены по спецификациям производителей 13.09.2026.
  const HARDWARE = [];
  const add = (group, name, memGB, bandwidthGBs, tflops, opts = {}) =>
    HARDWARE.push({ group, name, memGB, bandwidthGBs, tflops, unified: false, sysMemGB: 32, sysBandwidthGBs: 80, sysTflops: 1, ...opts });
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

  // --- Коробки на NVIDIA GB10: одна платформа, 128 ГБ LPDDR5X 273 ГБ/с, 1 PFLOP FP4 (≈60 TFLOPS FP16 практичных)
  const G_GB10 = 'Коробки NVIDIA GB10';
  for (const n of ['NVIDIA DGX Spark', 'ASUS Ascent GX10', 'Dell Pro Max with GB10', 'HP ZGX Nano AI Station G1n', 'Lenovo ThinkStation PGX',
                   'MSI EdgeXpert MS-C931', 'Acer Veriton GN100', 'Gigabyte AI TOP Atom'])
    uni(G_GB10, n, 273, 60, [128]);
  uni(G_GB10, '2× DGX Spark в связке (ConnectX-7)', 273, 60, [256], 256);

  // --- Коробки на AMD Strix Halo (Ryzen AI Max+ 395: 40 CU, 256 ГБ/с; GPU отдают до 96 из 128 ГБ)
  const G_SH = 'Коробки AMD Strix Halo';
  const sh = (n, mems, tf = 15, bw = 256) => uni(G_SH, n, bw, tf, mems, mems[mems.length - 1]);
  sh('Framework Desktop (AI Max+ 395)', [64, 128]);
  sh('Framework Desktop (AI Max 385, 32 CU)', [32], 12);
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

  // --- Мини-ПК и ноутбуки на других чипах (GPU встроенный, память общая)
  const G_MINI = 'Мини-ПК на других чипах';
  uni(G_MINI, 'Intel Core Ultra X9 388H Panther Lake (GMKtec EVO-T2 и др.)', 136, 15, [32, 64, 96, 128]);
  uni(G_MINI, 'Intel Core Ultra 9 285H Arrow Lake (ASUS NUC 15 Pro и др.)',  102, 8,  [32, 64, 96]);
  uni(G_MINI, 'AMD Ryzen AI 9 HX 370 (Beelink SER9, GEEKOM A9 Max, EVO-X1)',   90, 6,  [32, 64, 96]);
  uni(G_MINI, 'Qualcomm Snapdragon X2 Elite Extreme',                          228, 6,  [48, 64, 128]);
  uni(G_MINI, 'Qualcomm Snapdragon X2 Elite',                                  192, 5,  [32, 48, 64]);
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
  const cpu = (name, mem, bw, tf, mems) => add('Только процессор', name, mem, bw, tf, { unified: true, cpuOnly: true, memOptions: mems });
  cpu('Ноутбук, DDR4-3200 двухканал',      16,  51,  0.8, [8, 16, 32, 64]);
  cpu('ПК, DDR5-5600 двухканал',           32,  90,  1.5, [16, 32, 64, 96, 128, 192]);
  cpu('ПК, DDR5-6400 двухканал',           32,  102, 1.5, [16, 32, 64, 96, 128, 192]);
  cpu('Рабочая станция, DDR5 8-канал',     256, 300, 4,   [128, 256, 512, 1024]);
  cpu('Сервер, DDR5 12-канал (EPYC)',      512, 460, 6,   [256, 512, 1024, 2048]);

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

  function memoryNeeded({ model, bpw, ctx, kvBits }) {
    const weights = weightsBytes(model.params, bpw);
    const kv = kvBytesPerToken(model, kvBits, ctx) * ctx;
    const overhead = overheadBytes(weights, model, ctx);
    return { weights, kv, overhead, total: weights + kv + overhead };
  }

  // На каждый новый токен читаются активные веса и весь KV-кэш
  function bytesPerGeneratedToken({ model, bpw, ctx, kvBits }) {
    return weightsBytes(model.active, bpw) + kvBytesPerToken(model, kvBits, ctx) * ctx;
  }

  function genTokPerSec(bandwidthGBs, bytesPerToken) { return bandwidthGBs * 1e9 * BW_EFF / bytesPerToken; }
  function prefillTokPerSec(tflops, model) { return tflops * 1e12 * COMPUTE_EFF / (2 * model.active); }
  function requiredBandwidthGBs(bytesPerToken, targetTps) { return bytesPerToken * targetTps / BW_EFF / 1e9; }
  function requiredTflops(model, promptTokens, seconds) { return 2 * model.active * (promptTokens / seconds) / COMPUTE_EFF / 1e12; }

  // macOS по умолчанию отдаёт GPU 2/3 памяти при ≤36 ГБ и 3/4 выше (можно поднять через sysctl, но по умолчанию так)
  function usableMemoryBytes(hw) {
    const total = hw.memGB * GB;
    if (hw.cpuOnly) return total * OS_RESERVE_SYS;
    if (hw.unified) return total * (hw.memGB <= 36 ? 2 / 3 : 3 / 4);
    return total;
  }

  function fit(cfg, hw) {
    const mem = memoryNeeded(cfg);
    const usable = usableMemoryBytes(hw);
    const bytes = bytesPerGeneratedToken(cfg);
    const base = { need: mem.total, usable, mem, bytesPerToken: bytes };
    if (mem.total <= usable) {
      return { ...base, status: 'fits', gpuFraction: 1,
        genTps: genTokPerSec(hw.bandwidthGBs, bytes), prefillTps: prefillTokPerSec(hw.tflops, cfg.model) };
    }
    const sysUsable = hw.unified ? 0 : (hw.sysMemGB ?? 0) * GB * OS_RESERVE_SYS;
    if (mem.total <= usable + sysUsable) {
      const f = usable / mem.total;
      const tGen = f * bytes / (hw.bandwidthGBs * 1e9 * BW_EFF) + (1 - f) * bytes / (hw.sysBandwidthGBs * 1e9 * BW_EFF);
      const tPre = f / prefillTokPerSec(hw.tflops, cfg.model) + (1 - f) / prefillTokPerSec(hw.sysTflops ?? 1, cfg.model);
      return { ...base, status: 'offload', gpuFraction: f, genTps: 1 / tGen, prefillTps: 1 / tPre };
    }
    return { ...base, status: 'nofit', gpuFraction: 0, genTps: 0, prefillTps: 0 };
  }

  // Наибольший контекст, при котором всё влезает целиком в память ускорителя (двоичный поиск, память монотонна по ctx)
  function maxContext(cfg, hw) {
    let lo = 0, hi = 1 << 22;
    if (fit({ ...cfg, ctx: lo }, hw).status !== 'fits') return 0;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (fit({ ...cfg, ctx: mid }, hw).status === 'fits') lo = mid; else hi = mid;
    }
    return lo;
  }

  function verdict(f, goal) {
    if (f.status === 'nofit') return 'bad';
    const promptSec = goal.promptTokens / f.prefillTps;
    const genOk = f.genTps >= goal.genTps, preOk = promptSec <= goal.promptSeconds;
    if (f.status === 'fits' && genOk && preOk) return 'good';
    if (f.genTps >= goal.genTps / 2 && promptSec <= goal.promptSeconds * 3) return 'ok';
    return 'bad';
  }

  // Диагностика: во что упёрлись и что с этим делать. Возвращает числа, текст собирает интерфейс.
  function diagnose(cfg, hw, goal) {
    const f = fit(cfg, hw);
    const promptSec = f.prefillTps ? goal.promptTokens / f.prefillTps : Infinity;
    const kvShare = f.mem.kv / (f.mem.weights + f.mem.kv);
    const moe = cfg.model.active < cfg.model.params * 0.5;
    const byQuality = Object.keys(QUANTS).sort((a, b) => QUANTS[b].bpw - QUANTS[a].bpw);
    let bestQuantFits = null, bestQuantMeetsTarget = null;
    for (const k of byQuality) {
      const ff = fit({ ...cfg, bpw: QUANTS[k].bpw }, hw);
      if (ff.status !== 'fits') continue;
      if (!bestQuantFits) bestQuantFits = k;
      if (!bestQuantMeetsTarget && ff.genTps >= goal.genTps) bestQuantMeetsTarget = k;
    }
    const kvQ8Fits = cfg.kvBits > 8 && fit({ ...cfg, kvBits: 8 }, hw).status === 'fits';
    const kvQ4Fits = cfg.kvBits > 4 && fit({ ...cfg, kvBits: 4 }, hw).status === 'fits';
    // macOS по умолчанию режет память GPU; sysctl iogpu.wired_limit_mb поднимает её примерно до 90 %
    const macWiredLimitHelps = !!(hw.unified && !hw.cpuOnly && f.status !== 'fits' && f.need <= hw.memGB * GB * 0.9);
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
    else if (f.status === 'offload') bottleneck = 'offload';
    else if (f.genTps < goal.genTps) bottleneck = 'bandwidth';
    else if (promptSec > goal.promptSeconds) bottleneck = 'compute';
    else bottleneck = 'none';
    return { bottleneck, fit: f, promptSec, kvShare, moe, bestQuantFits, bestQuantMeetsTarget, kvQ8Fits, kvQ4Fits,
      macWiredLimitHelps, expertOffloadTps,
      requiredBandwidthGBs: requiredBandwidthGBs(f.bytesPerToken, goal.genTps),
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

  return { GB, BW_EFF, COMPUTE_EFF, QUANTS, KV_QUANTS, MODELS, HARDWARE,
    weightsBytes, kvBytesPerToken, memoryNeeded, bytesPerGeneratedToken, genTokPerSec, prefillTokPerSec,
    requiredBandwidthGBs, requiredTflops, usableMemoryBytes, fit, maxContext, verdict, diagnose, estimateLayers };
})();
