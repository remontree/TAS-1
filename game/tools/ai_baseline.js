// AI 기준선: 같은 노브 공간(엔진 413,343 × 기체)을 2단계로 탐색해 목표 고도·최고 고도 최고 기록을 찾는다.
// 기준 바람 4 m/s / 250°. 결과: data/ai_baseline.json
const fs = require('fs'), path = require('path'), F = require('../static/js/flight.js'), loadLUT = require('./lut.js');
const BASE = path.join(__dirname, '..');
const cfg = JSON.parse(fs.readFileSync(path.join(BASE, 'server_config.json')));
const L = loadLUT();
const K = L.meta.knobs, wind = { spd: 4, dir: 250 }, target = cfg.target_alt;
const t0 = Date.now();
function evalOne(k, a) {
  const row = L.row(k);
  const Dz = F.design(row, K.ox[+k[0]], a);
  if (!Dz.A.fit) return null;
  const fl = F.fly(Dz, { wind });
  return { k, a, apo: fl.apo, sc: F.score(Dz, fl, target) };   // 궤적은 버린다 (엔진 수십만 개)
}
const minBody = k => { for (let b = 0; b < 7; b++) { const row = L.row(k); if (F.airframe(row, K.ox[+k[0]], { body: b, fin: 2, nose: 3 }).fit) return b; } return null; };
// 1단계: 제약 통과 엔진 전부, 기본 기체
const first = [];
for (const k of L.keys()) {
  if ((L.row(k).codes || []).length) continue;
  const b = minBody(k); if (b == null) continue;
  const r = evalOne(k, { body: b, fin: 2, nose: 3 }); if (r) first.push(r);
}
const top = (arr, key, n) => arr.slice().sort((x, y) => y.sc[key] - x.sc[key]).slice(0, n);
const best = {};
for (const key of ['target', 'max']) {
  let b = null;
  for (const c of top(first, key, 50)) {
    const b0 = minBody(c.k);
    for (let body = b0; body < Math.min(7, b0 + 3); body++) for (let fin = 0; fin < 7; fin++) for (let nose = 0; nose < 4; nose++) {
      const r = evalOne(c.k, { body, fin, nose });
      if (r && (!b || r.sc[key] > b.sc[key])) b = r;
    }
  }
  best[key] = { score: b.sc[key], apo: b.apo, engine: b.k, air: b.a, note: `바람 ${wind.spd} m/s 기준` };
}
best.built_at = new Date().toISOString(); best.searched = first.length;
fs.writeFileSync(path.join(BASE, 'data', 'ai_baseline.json'), JSON.stringify(best, null, 1));
console.log(JSON.stringify(best), ((Date.now() - t0) / 1000).toFixed(1) + 's');
