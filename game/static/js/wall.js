(function () {
  'use strict';
  const F = window.Flight, Rn = window.Render, fmt = Rn.fmt;
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  let sky = null, lastReplay = 0, queue = [], playing = false, prevTop = {};
  try { sky = Rn.Sky($('gl'), {}); } catch (_) { }
  const get = p => fetch(p, { credentials: 'same-origin' }).then(r => r.json());

  setInterval(() => { $('s-clock').textContent = new Date().toLocaleTimeString('ko-KR', { hour12: false }); }, 1000);

  function boardHTML(rows, mode, ai) {
    const h = rows.slice(0, 10).map(r => `<div class="lb-row ${r.rank <= 3 ? 'top' : ''} ${prevTop[mode] && !prevTop[mode].includes(r.pid) ? 'fresh' : ''}"><span class="r">${String(r.rank).padStart(2, '0')}</span><span class="n"><b>${esc(r.name)}</b><small>${esc(r.dept)}</small></span><span class="h">${fmt(r.apo)} m</span><span class="s">${mode === 'target' ? fmt(r.score) : fmt(r.score) + 'm'}</span></div>`);
    if (ai && ai[mode]) {
      const a = ai[mode], at = rows.filter(r => r.score > a.score).length;
      if (at < 10) h.splice(at, 0, `<div class="lb-row ai"><span class="r">AI</span><span class="n"><b>TAS-1 MDO</b><small>최적화기 기준선</small></span><span class="h">${fmt(a.apo)} m</span><span class="s">${mode === 'target' ? fmt(a.score) : fmt(a.score) + 'm'}</span></div>`);
    }
    prevTop[mode] = rows.slice(0, 10).map(r => r.pid);
    return h.join('') || '<div class="empty">아직 공식 기록이 없습니다</div>';
  }
  async function boards() {
    try {
      const [t, m] = await Promise.all([get('/api/board?mode=target'), get('/api/board?mode=max')]);
      $('tgt').textContent = fmt(t.target);
      $('lb-target').innerHTML = boardHTML(t.rows, 'target', t.ai);
      $('lb-max').innerHTML = boardHTML(m.rows, 'max', m.ai);
      if (t.ai && t.ai.target) $('ai').innerHTML = `<b style="font-size:1.1em">AI 기준선</b><span style="color:var(--info)">목표 ${fmt(t.ai.target.score)}점 · 최고 ${fmt(t.ai.max.score)} m</span><span>TAS-1 최적화기가 같은 노브 공간을 탐색한 결과</span>`;
    } catch (_) { }
  }
  async function feed() {
    try {
      const d = await get('/api/feed');
      const s = d.stats; $('s-players').textContent = fmt(s.players); $('s-launches').textContent = fmt(s.launches); $('s-crash').textContent = fmt(s.crashes); $('s-10m').textContent = fmt(s.launches_10m);
      $('s-wind').textContent = `바람 ${d.wind.spd} m/s · ${d.wind.dir}°`;
      $('feed').innerHTML = d.feed.slice(0, 7).map(f => { const ok = f.outcome === 'APOGEE'; return `<div class="${ok ? '' : 'bad'}"><span>${new Date(f.t * 1000).toLocaleTimeString('ko-KR', { hour12: false })}</span> ${esc(f.name)} · ${f.official ? '공식' : '연습'} · ${ok ? fmt(f.apo) + ' m' + (f.official ? ' · ' + fmt(f.score) + '점' : '') : f.outcome}</div>`; }).join('') || '<div>아직 발사가 없습니다</div>';
      $('rud').innerHTML = d.rud.map((r, i) => `<div>${String(i + 1).padStart(2, '0')} · ${esc(r.name)} · ${r.outcome === 'BURST' ? '발사대 파열' : '충돌 ' + r.impact_v + ' m/s'} · 핀 ${fmt(F.AIR.fin.vals[r.fin] * 1000)} mm</div>`).join('') || '<div>아직 추락이 없습니다</div>';
      Rn.drawSide($('side'), d.feed.filter(f => f.path).slice(0, 6).reverse().map((f, i, a) => ({ pts: f.path, bad: f.outcome !== 'APOGEE', done: true, alpha: i === a.length - 1 ? 1 : 0.4, lw: i === a.length - 1 ? 2.2 : 1.2 })), { target: d.target });
      const latest = d.feed.find(f => f.official);
      if (latest && latest.id !== lastReplay) { lastReplay = latest.id; queue = [latest.id]; next(); }
    } catch (_) { }
  }
  async function next() {
    if (playing || !queue.length || !sky) return;
    playing = true;
    const id = queue.shift();
    try {
      const r = await get('/api/replay/' + id);
      const Dz = F.design(r.row, r.ox, r.air);
      const fl = { tr: r.tr, outcome: r.outcome, apo: r.apo, ev: evFrom(r.tr, r.outcome), land: null, wind: r.wind, chug: (r.row.codes || []).includes('ERR_INJECTOR_DP'), maxAoa: 0 };
      const last = r.tr[r.tr.length - 1]; if (r.outcome === 'APOGEE') fl.land = [last.p[0], last.p[1]];
      $('r-who').textContent = r.name;
      sky.load(Dz, fl, r.wind);
      sky.start((ts, s) => {
        $('h-alt').textContent = fmt(s.h); $('h-spd').textContent = fmt(s.v * 3.6); $('h-clock').textContent = 'T+' + fmt(Math.max(0, ts), 1);
        banner(ts < 0 ? 'idle' : 'idle', ts < 0 ? 'T−' + Math.ceil(-ts * 2) : (s.chute ? 'UNDER CANOPY' : 'ASCENT'), ts < 0 ? '카운트다운' : `고도 ${fmt(s.h)} m`);
      }, () => {
        const ok = r.outcome === 'APOGEE';
        banner(ok ? 'go' : 'nogo', ok ? 'APOGEE ' + fmt(r.apo) + ' m' : r.outcome, ok ? '낙하산 착지' : '오늘의 추락 후보');
        playing = false; setTimeout(next, 500);
      });
    } catch (_) { playing = false; }
  }
  function evFrom(tr, outcome) {
    const ev = {}; let mx = 0, mi = 0;
    tr.forEach((p, i) => { if (!p.chute && p.h > mx) { mx = p.h; mi = i; } });
    if (outcome === 'APOGEE') ev.apo = tr[mi].t; else ev.impact = tr[tr.length - 1].t;
    return ev;
  }
  function banner(cls, t, s) { const b = $('banner'); b.className = 'banner ' + cls; b.innerHTML = `${t}<small>${s}</small>`; }
  get('/api/url').then(d => ($('url').textContent = d.url)).catch(() => { });
  get('/api/meta').then(m => ($('lut').textContent = `엔진 성능: TAS-1 시계열 서로게이트 · ${fmt(m.lut.n)}개 설계를 ${m.lut.inference_s}초에 계산 (${m.lut.device.toUpperCase()})`)).catch(() => { });
  boards(); feed();
  setInterval(boards, 5000); setInterval(feed, 3000);
  addEventListener('resize', feed);
})();
