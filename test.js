// Тесты формул калькулятора. Запуск: node test.js
// Формулы живут в index.html внутри <script id="calc">, чтобы страница была одним файлом.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(__dirname + '/index.html', 'utf8');
const m = html.match(/<script id="calc">([\s\S]*?)<\/script>/);
if (!m) throw new Error('В index.html нет блока <script id="calc">');
const ctx = { console };
vm.runInNewContext(m[1], ctx);
const C = ctx.LLMCalc;

const GB = 1024 ** 3;
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} ≉ ${b} (±${tol})`);

const tests = [];
const test = (name, fn) => tests.push({ name, fn });

// Модели-стенды
const llama8b = { params: 8.03e9, active: 8.03e9, layers: 32, kvHeads: 8, headDim: 128, hidden: 4096 };
const deepseek = { params: 671e9, active: 37e9, layers: 61, kvHeads: 1, headDim: 576, hidden: 7168, kvFactor: 1 };
const gemma27 = { params: 27e9, active: 27e9, layers: 62, kvHeads: 16, headDim: 128, hidden: 5376, localRatio: 5 / 6, window: 1024 };

test('веса: параметры × бит на вес / 8', () => {
  near(C.weightsBytes(8e9, 4.85), 4.85e9, 1, 'Q4_K_M 8B');
  near(C.weightsBytes(8e9, 16), 16e9, 1, 'FP16 8B');
});

test('KV на токен для плотной модели с GQA', () => {
  // 2 × 32 слоя × 8 голов × 128 × 2 байта = 131072 байт
  assert.strictEqual(C.kvBytesPerToken(llama8b, 16, 8192), 131072);
  assert.strictEqual(C.kvBytesPerToken(llama8b, 8, 8192), 65536);
});

test('KV на токен для MLA (DeepSeek): один вектор на слой, не пара', () => {
  // 1 × 61 × 1 × 576 × 2 = 70272
  assert.strictEqual(C.kvBytesPerToken(deepseek, 16, 8192), 70272);
});

test('KV для скользящего окна: локальные слои хранят не больше окна', () => {
  const full = 2 * 62 * 16 * 128 * 2; // 507904 на токен без окна
  // при ctx 8192 глобальные 1/6 слоёв держат весь контекст, локальные 5/6 — только 1024
  const expected = full * (1 / 6 + (5 / 6) * (1024 / 8192));
  near(C.kvBytesPerToken(gemma27, 16, 8192), expected, 1, 'gemma окно');
  // при ctx меньше окна — как без окна
  near(C.kvBytesPerToken(gemma27, 16, 512), full, 1, 'gemma малый ctx');
});

test('KV только в части слоёв (гибриды: Qwen 3.5+, Nemotron 3, Kimi K3)', () => {
  const hybrid = { ...llama8b, layers: 64, kvLayers: 16 };
  // 2 × 16 × 8 × 128 × 2 = 65536
  assert.strictEqual(C.kvBytesPerToken(hybrid, 16, 8192), 65536);
});

test('KV со сжатием (DeepSeek V4: 10 % от обычного MLA)', () => {
  const v4 = { ...deepseek, kvScale: 0.1 };
  near(C.kvBytesPerToken(v4, 16, 8192), 7027.2, 0.01, 'v4');
});

test('полная память: веса + KV + накладные', () => {
  const r = C.memoryNeeded({ model: llama8b, bpw: 4.85, ctx: 8192, kvBits: 16 });
  near(r.weights, 8.03e9 * 4.85 / 8, 1, 'веса');
  near(r.kv, 131072 * 8192, 1, 'kv');
  assert.ok(r.overhead > 0.5 * GB, 'накладные не меньше полугигабайта');
  near(r.total, r.weights + r.kv + r.overhead, 1, 'сумма');
});

test('скорость генерации ограничена пропускной способностью', () => {
  // 1000 ГБ/с × 0.7 / (8B × 4.85 бит + KV 8192 токенов) ≈ 117 ток/с
  const bytes = C.bytesPerGeneratedToken({ model: llama8b, bpw: 4.85, ctx: 8192, kvBits: 16 });
  near(bytes, 8.03e9 * 4.85 / 8 + 131072 * 8192, 1, 'байт на токен');
  near(C.genTokPerSec(1000, bytes), 1000e9 * 0.7 / bytes, 0.01, 'tps');
});

test('MoE: скорость считается по активным параметрам', () => {
  const bytes = C.bytesPerGeneratedToken({ model: deepseek, bpw: 4.85, ctx: 0, kvBits: 16 });
  near(bytes, 37e9 * 4.85 / 8, 1, 'активные байты');
});

test('обработка промпта ограничена вычислениями', () => {
  // 165 TFLOPS × 0.5 / (2 × 8.03e9) ≈ 5137 ток/с
  near(C.prefillTokPerSec(165, llama8b), 165e12 * 0.5 / (2 * 8.03e9), 1, 'prefill');
});

test('требуемая пропускная способность и TFLOPS под цель', () => {
  near(C.requiredBandwidthGBs(5e9, 10), 5e9 * 10 / 0.7 / 1e9, 0.01, 'ГБ/с');
  // 4096 токенов за 10 с при 8B: 2 × 8e9 × 409.6 / 0.5 = 13.1 TFLOPS
  near(C.requiredTflops(llama8b, 4096, 10), 2 * 8.03e9 * 409.6 / 0.5 / 1e12, 0.01, 'TFLOPS');
});

test('доступная память: Mac отдаёт GPU 2/3 до 36 ГБ и 3/4 выше', () => {
  near(C.usableMemoryBytes({ unified: true, memGB: 16 }), 16 * GB * 2 / 3, 1, '16 ГБ');
  near(C.usableMemoryBytes({ unified: true, memGB: 64 }), 64 * GB * 3 / 4, 1, '64 ГБ');
  near(C.usableMemoryBytes({ unified: false, memGB: 24 }), 24 * GB, 1, 'дискретная');
});

test('влезает целиком — скорость по памяти ускорителя', () => {
  const hw = { unified: false, memGB: 24, bandwidthGBs: 1000, tflops: 165, sysMemGB: 32, sysBandwidthGBs: 80 };
  const cfg = { model: llama8b, bpw: 4.85, ctx: 8192, kvBits: 16 };
  const f = C.fit(cfg, hw);
  assert.strictEqual(f.status, 'fits');
  near(f.gpuFraction, 1, 1e-9, 'доля на GPU');
  near(f.genTps, C.genTokPerSec(1000, C.bytesPerGeneratedToken(cfg)), 0.01, 'tps');
});

test('не влезает в VRAM, но влезает в VRAM+ОЗУ — частичная выгрузка', () => {
  const hw = { unified: false, memGB: 8, bandwidthGBs: 1000, tflops: 165, sysMemGB: 32, sysBandwidthGBs: 80 };
  const cfg = { model: llama8b, bpw: 8.5, ctx: 4096, kvBits: 16 }; // ~8.5 ГБ весов
  const f = C.fit(cfg, hw);
  assert.strictEqual(f.status, 'offload');
  assert.ok(f.gpuFraction > 0 && f.gpuFraction < 1, 'часть на GPU');
  // время на токен = доля_gpu × байт / bw_gpu + доля_cpu × байт / bw_cpu
  const bytes = C.bytesPerGeneratedToken(cfg);
  const t = f.gpuFraction * bytes / (1000e9 * 0.7) + (1 - f.gpuFraction) * bytes / (80e9 * 0.7);
  near(f.genTps, 1 / t, 0.01, 'tps при выгрузке');
  const onlyGpu = C.genTokPerSec(1000, bytes), onlySys = C.genTokPerSec(80, bytes);
  assert.ok(f.genTps < onlyGpu && f.genTps > onlySys, 'между чистым GPU и чистой ОЗУ');
});

test('не влезает никуда', () => {
  const hw = { unified: false, memGB: 8, bandwidthGBs: 1000, tflops: 165, sysMemGB: 8, sysBandwidthGBs: 80 };
  const f = C.fit({ model: llama8b, bpw: 16, ctx: 32768, kvBits: 16 }, hw); // ~22 ГБ против 8 + 8×0.85
  assert.strictEqual(f.status, 'nofit');
});

test('единая память: нет выгрузки, либо влезает, либо нет', () => {
  const hw = { unified: true, memGB: 16, bandwidthGBs: 120, tflops: 4.6 };
  assert.strictEqual(C.fit({ model: llama8b, bpw: 4.85, ctx: 8192, kvBits: 16 }, hw).status, 'fits');
  assert.strictEqual(C.fit({ model: llama8b, bpw: 16, ctx: 8192, kvBits: 16 }, hw).status, 'nofit');
});

test('максимальный контекст, который поместится', () => {
  const hw = { unified: false, memGB: 24, bandwidthGBs: 1000, tflops: 165, sysMemGB: 32, sysBandwidthGBs: 80 };
  const cfg = { model: llama8b, bpw: 4.85, kvBits: 16 };
  const mx = C.maxContext(cfg, hw);
  assert.ok(mx > 100000, `ожидали >100k, получили ${mx}`);
  // при этом контексте память ещё влезает, а при +10% — нет
  assert.strictEqual(C.fit({ ...cfg, ctx: mx }, hw).status, 'fits');
  assert.notStrictEqual(C.fit({ ...cfg, ctx: Math.round(mx * 1.1) }, hw).status, 'fits');
});

test('вердикт: удобно / терпимо / не годится', () => {
  const goal = { genTps: 10, promptTokens: 4096, promptSeconds: 10 };
  assert.strictEqual(C.verdict({ status: 'fits', genTps: 50, prefillTps: 2000 }, goal), 'good');
  assert.strictEqual(C.verdict({ status: 'fits', genTps: 6, prefillTps: 2000 }, goal), 'ok');
  assert.strictEqual(C.verdict({ status: 'fits', genTps: 50, prefillTps: 200 }, goal), 'ok'); // 20 с на промпт
  assert.strictEqual(C.verdict({ status: 'offload', genTps: 3, prefillTps: 100 }, goal), 'bad');
  assert.strictEqual(C.verdict({ status: 'nofit' }, goal), 'bad');
});

// --- диагностика узкого места и рекомендации
const rtx4090 = { unified: false, memGB: 24, bandwidthGBs: 1008, tflops: 165, sysMemGB: 64, sysBandwidthGBs: 90, sysTflops: 1 };
const goal = { genTps: 10, promptTokens: 4096, promptSeconds: 10 };

test('диагностика: не влезает — узкое место память, подсказан самый качественный влезающий квант', () => {
  const d = C.diagnose({ model: llama8b, bpw: 16, ctx: 8192, kvBits: 16 }, { ...rtx4090, memGB: 12, sysMemGB: 4 }, goal);
  assert.strictEqual(d.bottleneck, 'memory');
  // FP16 (16 ГБ) не влезает в 12, Q8_0 (8.5 ГБ) — влезает
  assert.strictEqual(d.bestQuantFits, 'Q8_0');
  assert.ok(d.bestQuantMeetsTarget === 'Q8_0', 'Q8 и в цель по скорости попадает');
});

test('диагностика: ничего не влезает — bestQuantFits пустой', () => {
  const d = C.diagnose({ model: { ...llama8b, params: 405e9, active: 405e9 }, bpw: 4.85, ctx: 8192, kvBits: 16 }, { ...rtx4090, sysMemGB: 16 }, goal);
  assert.strictEqual(d.bottleneck, 'memory');
  assert.strictEqual(d.bestQuantFits, null);
});

test('диагностика: MoE с выгрузкой — скорость упирается в ОЗУ по активным весам и остаётся пригодной', () => {
  const qwen235 = { params: 235e9, active: 22e9, layers: 94, kvHeads: 4, headDim: 128, hidden: 4096 };
  const cfg = { model: qwen235, bpw: 4.85, ctx: 8192, kvBits: 16 };
  const hw = { ...rtx4090, sysMemGB: 192 };
  assert.strictEqual(C.fit(cfg, hw).status, 'offload');
  const d = C.diagnose(cfg, hw, goal);
  assert.strictEqual(d.bottleneck, 'offload');
  // ≈ 90 ГБ/с × 0.7 ÷ (22B × 4.85/8) с поправкой на KV с GPU
  near(d.expertOffloadTps, 90e9 * 0.7 / (22e9 * 4.85 / 8), 0.2, 'формула');
  // плотная модель того же размера при той же выгрузке была бы на порядок медленнее
  const dense = C.fit({ ...cfg, model: { ...qwen235, active: 235e9 } }, hw);
  assert.ok(d.expertOffloadTps > 5 * dense.genTps, `${d.expertOffloadTps} против плотной ${dense.genTps}`);
});

test('диагностика: плотная модель с выгрузкой — совета про экспертов нет', () => {
  const d = C.diagnose({ model: llama8b, bpw: 8.5, ctx: 4096, kvBits: 16 }, { ...rtx4090, memGB: 8 }, goal);
  assert.strictEqual(d.bottleneck, 'offload');
  assert.strictEqual(d.expertOffloadTps, null);
});

test('диагностика: влезает, но медленно — пропускная способность; подсказан квант под цель', () => {
  const l70 = { params: 70.6e9, active: 70.6e9, layers: 80, kvHeads: 8, headDim: 128, hidden: 8192 };
  const m4pro = { unified: true, memGB: 64, bandwidthGBs: 273, tflops: 9.2 };
  const d = C.diagnose({ model: l70, bpw: 4.85, ctx: 8192, kvBits: 16 }, m4pro, { ...goal, genTps: 5.5 });
  assert.strictEqual(d.bottleneck, 'bandwidth');
  // Q4_K_M даёт ~4.2 ток/с, Q3_K_M ~5.1, Q2_K ~5.9 → под цель 5.5 подходит Q2_K
  assert.strictEqual(d.bestQuantMeetsTarget, 'Q2_K');
});

test('диагностика: генерация в норме, промпт медленный — вычисления', () => {
  const l70 = { params: 70.6e9, active: 70.6e9, layers: 80, kvHeads: 8, headDim: 128, hidden: 8192 };
  const m2ultra = { unified: true, memGB: 192, bandwidthGBs: 800, tflops: 27 };
  const d = C.diagnose({ model: l70, bpw: 4.85, ctx: 8192, kvBits: 16 }, m2ultra, goal);
  assert.strictEqual(d.bottleneck, 'compute');
  assert.ok(d.promptSec > goal.promptSeconds);
});

test('диагностика: всё в норме — узкого места нет, есть запас по качеству', () => {
  const d = C.diagnose({ model: llama8b, bpw: 4.85, ctx: 8192, kvBits: 16 }, rtx4090, goal);
  assert.strictEqual(d.bottleneck, 'none');
  assert.strictEqual(d.bestQuantMeetsTarget, 'FP16', 'на 4090 8B влезает даже FP16 и держит цель');
  assert.ok(d.kvShare < 0.5, 'KV не доминирует');
});

test('диагностика: Mac — совет поднять лимит GPU, если модель влезает в 90 % памяти', () => {
  const m4 = { unified: true, memGB: 16, bandwidthGBs: 120, tflops: 4.6 };
  // 8B Q8 при 32k: ~14.2 ГБ; лимит по умолчанию 2/3 от 16 = 10.7 ГБ — не влезает, а в 90 % (14.4 ГБ) влезает
  const d = C.diagnose({ model: llama8b, bpw: 8.5, ctx: 32768, kvBits: 16 }, m4, goal);
  assert.strictEqual(d.bottleneck, 'memory');
  assert.strictEqual(d.macWiredLimitHelps, true);
  const d2 = C.diagnose({ model: llama8b, bpw: 16, ctx: 8192, kvBits: 16 }, m4, goal);
  assert.strictEqual(d2.macWiredLimitHelps, false);
});

test('диагностика: KV Q8 спасает, когда не хватает чуть-чуть', () => {
  // 8B Q4 ctx 128k: веса 4.9 ГБ + KV 16 ГБ = не влезает в 16 ГБ, с KV Q8 (8 ГБ) — влезает
  const hw = { ...rtx4090, memGB: 16, sysMemGB: 4 };
  const d = C.diagnose({ model: llama8b, bpw: 4.85, ctx: 131072, kvBits: 16 }, hw, goal);
  assert.strictEqual(d.bottleneck, 'memory');
  assert.strictEqual(d.kvQ8Fits, true);
  assert.ok(d.kvShare > 0.5, 'KV доминирует');
});

test('оценка слоёв для своей модели по числу параметров', () => {
  assert.strictEqual(C.estimateLayers(8e9), 32);
  assert.strictEqual(C.estimateLayers(70e9), 80);
  const mid = C.estimateLayers(30e9);
  assert.ok(mid > 32 && mid < 80, `30B → ${mid}`);
});

test('пресеты моделей и железа заполнены', () => {
  assert.ok(C.MODELS.length >= 110, `моделей мало: ${C.MODELS.length}`);
  assert.ok(C.HARDWARE.length >= 80, `железа мало: ${C.HARDWARE.length}`);
  for (const m of C.MODELS) {
    for (const k of ['family', 'name', 'params', 'active', 'layers', 'kvHeads', 'headDim', 'hidden', 'maxCtx']) assert.ok(m[k] !== undefined, `${m.name}: нет ${k}`);
    assert.ok(m.active <= m.params, `${m.name}: активных больше общих`);
    assert.ok(m.layers >= 8 && m.layers <= 160, `${m.name}: странное число слоёв ${m.layers}`);
    if (m.kvLayers !== undefined) assert.ok(m.kvLayers > 0 && m.kvLayers <= m.layers, `${m.name}: kvLayers вне диапазона`);
    if (m.kvScale !== undefined) assert.ok(m.kvScale > 0 && m.kvScale <= 1, `${m.name}: kvScale вне диапазона`);
    assert.ok(m.kvHeads * m.headDim <= m.hidden * 2, `${m.name}: KV шире скрытого слоя`);
  }
  const names = C.MODELS.map(m => m.name);
  assert.strictEqual(new Set(names).size, names.length, 'имена моделей повторяются');
  for (const h of C.HARDWARE) {
    for (const k of ['name', 'group', 'memGB', 'bandwidthGBs', 'tflops', 'unified']) assert.ok(h[k] !== undefined, `${h.name}: нет ${k}`);
    if (h.unified) assert.ok(Array.isArray(h.memOptions) && h.memOptions.includes(h.memGB), `${h.name}: у единой памяти нет вариантов объёма`);
    assert.ok(h.bandwidthGBs > 0 && h.bandwidthGBs < 10000 && h.tflops > 0, `${h.name}: странные числа`);
  }
  const hnames = C.HARDWARE.map(h => h.name);
  assert.strictEqual(new Set(hnames).size, hnames.length, 'имена железа повторяются');
  for (const g of ['Apple · текущие Mac', 'Коробки NVIDIA GB10', 'Коробки AMD Strix Halo', 'NVIDIA Jetson']) assert.ok(C.HARDWARE.some(h => h.group === g), `нет группы ${g}`);
  assert.ok(C.QUANTS.Q4_K_M.bpw > 4 && C.QUANTS.Q4_K_M.bpw < 5.5, 'Q4_K_M ≈ 4.85 бит');
});

let failed = 0;
for (const t of tests) {
  try { t.fn(); console.log('  ok   ' + t.name); }
  catch (e) { failed++; console.log('  FAIL ' + t.name + '\n       ' + e.message); }
}
console.log(failed ? `\n${failed} из ${tests.length} упало` : `\nвсе ${tests.length} прошли`);
process.exit(failed ? 1 : 0);
