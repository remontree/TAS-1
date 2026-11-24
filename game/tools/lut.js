// engine_lut.jsonl.gz 읽기: 첫 줄 meta, 이후 "키<TAB>행 JSON". 행은 문자열로 들고 있다가 필요할 때 파싱 (41만 행)
const zlib = require('zlib'), fs = require('fs'), path = require('path');
module.exports = function loadLUT() {
  const buf = zlib.gunzipSync(fs.readFileSync(path.join(__dirname, '..', 'data', 'engine_lut.jsonl.gz')));
  const raw = new Map();
  let meta = null, pos = 0;
  while (pos < buf.length) {
    let nl = buf.indexOf(10, pos); if (nl < 0) nl = buf.length;
    const line = buf.toString('utf8', pos, nl); pos = nl + 1;
    if (!line) continue;
    if (!meta) { meta = JSON.parse(line); continue; }
    const t = line.indexOf('\t'); raw.set(line.slice(0, t), line.slice(t + 1));
  }
  return { meta, keys: () => raw.keys(), row: k => { const r = JSON.parse(raw.get(k)); r.wall_t = meta.wall_t; return r; } };
};
