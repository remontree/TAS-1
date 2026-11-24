// 서버 재계산: stdin {row, ox, air, wind, target} → stdout {fl, score, margin}
const F = require('../static/js/flight.js');
let buf = '';
process.stdin.on('data', d => (buf += d));
process.stdin.on('end', () => {
  const q = JSON.parse(buf);
  const Dz = F.design(q.row, q.ox, q.air);
  const fl = F.fly(Dz, { wind: q.wind });
  const score = F.score(Dz, fl, q.target);
  process.stdout.write(JSON.stringify({ fl, score, margin: +Dz.A.margin.toFixed(3), errs: Dz.errs, warns: Dz.warns }));
});
