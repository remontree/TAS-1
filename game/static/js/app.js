(function () {
  'use strict';
  const F = window.Flight, Rn = window.Render, C = Rn.C, fmt = Rn.fmt;
  const $ = id => document.getElementById(id);
  const ENG = [
    ['ox', 'OX Load', '산화제 · 배터리 용량', 'kg', 1, 'tank'],
    ['p', 'Pressure', '탱크 압력 · 오버클럭', 'MPa', 1, 'tank'],
    ['n', 'Injector', '분사구 · 대역폭', '구', 0, 'inj'],
    ['L', 'Fuel Grain', '연료 길이 · 배터리 셀', 'm', 2, 'grain'],
    ['g', 'Grain Ports', '연료 포트 수 · 코어 수', '개', 0, 'grain'],
    ['e', 'Nozzle', '확장비 · 출력 포트', 'ε', 1, 'nozzle'],
  ];
  const KEYS = { ox: 'ox', p: 'p_MPa', n: 'n', L: 'fuel_m', e: 'eps', g: 'ports' };
  const DEF = { ox: 4, p: 4, n: 4, L: 4, e: 4, g: 1, body: 4, fin: 4, nose: 1 };
  let META = null, ME = null, st = Object.assign({}, DEF), active = 'tank', mode = 'target';
  let cur = null, curRow = null, sky = null, lastLaunch = null, reqSeq = 0;
  const cache = new Map();

  try { const s = JSON.parse(localStorage.getItem('tas1-design3') || 'null'); if (s) Object.assign(st, s); } catch (_) { }  // 단계 수가 바뀌어 새 키 사용
  const save = () => { try { localStorage.setItem('tas1-design3', JSON.stringify(st)); } catch (_) { } };

  async function api(path, body) {
    const opt = { credentials: 'same-origin', headers: {} };
    if (body !== undefined) { opt.method = 'POST'; opt.headers['Content-Type'] = 'application/json'; opt.headers['X-TAS1'] = '1'; opt.body = JSON.stringify(body); }
    const r = await fetch(path, opt);
    let j = null; try { j = await r.json(); } catch (_) { }
    if (!r.ok) { const e = new Error((j && j.error) || '네트워크 오류가 발생했습니다.'); e.status = r.status; e.data = j; throw e; }
    return j;
  }
  let toastT;
  function toast(msg) { const t = $('toast'); t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => (t.hidden = true), 3200); }

  // ── auth
  function showAuth() { $('auth').hidden = false; $('app').hidden = true; }
  $('tab-reg').onclick = () => { $('tab-reg').setAttribute('aria-pressed', 'true'); $('tab-login').setAttribute('aria-pressed', 'false'); $('f-reg').hidden = false; $('f-login').hidden = true; };
  $('tab-login').onclick = () => { $('tab-login').setAttribute('aria-pressed', 'true'); $('tab-reg').setAttribute('aria-pressed', 'false'); $('f-login').hidden = false; $('f-reg').hidden = true; };
  $('f-reg').onsubmit = async e => {
    e.preventDefault(); $('reg-err').textContent = '';
    const body = { name: $('r-name').value, student_id: $('r-sid').value, dept: $('r-dept').value, phone: $('r-phone').value, consent: $('r-consent').checked };
    if (!body.consent) { $('reg-err').textContent = '개인정보 수집·이용에 동의해 주세요.'; return; }
    $('reg-btn').disabled = true;
    try { await api('/api/register', body); await enter(); }
    catch (er) { $('reg-err').textContent = er.message; if (er.data && er.data.code === 'exists') { $('tab-login').onclick(); $('l-sid').value = body.student_id; } }
    finally { $('reg-btn').disabled = false; }
  };
  $('f-login').onsubmit = async e => {
    e.preventDefault(); $('login-err').textContent = '';
    try { await api('/api/login', { student_id: $('l-sid').value, phone: $('l-phone').value }); await enter(); }
    catch (er) { $('login-err').textContent = er.message; }
  };
  $('logout').onclick = async () => { await api('/api/logout', {}).catch(() => { }); location.reload(); };

  // ── knobs
  function knobEl(id, nm, an, u, d, view, vals) {
    const w = document.createElement('div'); w.className = 'knob'; w.dataset.view = view;
    w.innerHTML = `<div class="row"><span><span class="nm">${nm}</span> <span class="an">${an}</span></span><span class="v" id="v-${id}"></span></div><input type="range" min="0" max="${vals.length - 1}" step="1" id="k-${id}" aria-label="${nm}">`;
    const inp = w.querySelector('input');
    inp.addEventListener('pointerdown', () => { active = view; });
    inp.addEventListener('input', () => { st[id] = +inp.value; active = view; save(); update(); });
    w._vals = vals; w._u = u; w._d = d; w._id = id;
    return w;
  }
  function buildKnobs() {
    const K = META.knobs;
    ENG.forEach(([id, nm, an, u, d, view]) => $('k-eng').appendChild(knobEl(id, nm, an, u, d, view, K[KEYS[id]])));
    const air = $('k-air'), nose = $('knob-nose');
    air.insertBefore(knobEl('body', 'Body Tube', '지관 길이 · 케이스', 'm', 1, 'body', F.AIR.body.vals), nose);
    air.insertBefore(knobEl('fin', 'Fin Span', '핀 크기 · 방향키', 'mm', 0, 'fin', F.AIR.fin.vals.map(v => v * 1000)), nose);
    F.NOSES.forEach((n, i) => {
      const b = document.createElement('button'); b.type = 'button';
      let d = 'M2 10'; for (let j = 0; j <= 20; j++) d += ` L${(2 + j / 20 * 36).toFixed(1)} ${(10 - F.noseR(n.id, j / 20 * F.LN) / F.R * 7).toFixed(1)}`;
      for (let j = 20; j >= 0; j--) d += ` L${(2 + j / 20 * 36).toFixed(1)} ${(10 + F.noseR(n.id, j / 20 * F.LN) / F.R * 7).toFixed(1)}`;
      b.innerHTML = `<svg viewBox="0 0 40 20" aria-hidden="true"><path d="${d} Z" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>${n.nm}`;
      b.onclick = () => { st.nose = i; active = 'nose'; save(); update(); };
      $('noses').appendChild(b);
    });
    $('t-eng').onclick = () => tab('eng'); $('t-air').onclick = () => tab('air');
    const z = [[-2, 0, C.nogo], [0, 1, C.warn], [1, 3.2, C.go], [3.2, 5, C.warn]];  // Carbon support 색
    $('zones').innerHTML = z.map(([a, b, c]) => `<i style="width:${(b - a) / 7 * 100}%;background:${c}"></i>`).join('');
    $('ticks').innerHTML = [-2, 0, 1, 2, 3, 5].map(v => `<span style="left:${mpos(v)}%">${v}</span>`).join('');
  }
  const mpos = m => Math.max(0, Math.min(1, (m + 2) / 7)) * 100;
  function tab(t) { $('t-eng').setAttribute('aria-selected', t === 'eng'); $('t-air').setAttribute('aria-selected', t === 'air'); $('k-eng').hidden = t !== 'eng'; $('k-air').hidden = t !== 'air'; active = t === 'eng' ? 'tank' : 'fin'; if (cur) render(); }

  const engKey = () => '' + st.ox + st.p + st.n + st.L + st.e + st.g;
  async function getRow(key) {
    if (cache.has(key)) return cache.get(key);
    const row = await api('/api/engine/' + key); cache.set(key, row); return row;
  }
  let upT;
  function update() {
    document.querySelectorAll('.knob').forEach(k => {
      if (!k._id) return; const inp = $('k-' + k._id); inp.value = st[k._id]; inp.style.setProperty('--p', (st[k._id] / (k._vals.length - 1) * 100) + '%');
      $('v-' + k._id).textContent = fmt(k._vals[st[k._id]], k._d) + ' ' + k._u;
    });
    $('v-nose').textContent = F.NOSES[st.nose].ko;
    document.querySelectorAll('#noses button').forEach((b, i) => b.setAttribute('aria-pressed', i === st.nose));
    const key = engKey(), seq = ++reqSeq;
    if (cache.has(key)) { curRow = cache.get(key); render(); return; }
    $('c-eng').classList.add('loading');
    clearTimeout(upT);
    upT = setTimeout(async () => {
      try { const row = await getRow(key); if (seq === reqSeq) { curRow = row; render(); } }
      catch (e) { toast(e.message); if (e.status === 401) showAuth(); }
      finally { $('c-eng').classList.remove('loading'); }
    }, 90);
  }
  function render() {
    if (!curRow) return;
    const ox = META.knobs.ox[st.ox];
    cur = F.design(curRow, ox, { body: st.body, fin: st.fin, nose: st.nose });
    const A = cur.A, row = curRow;
    document.querySelectorAll('.knob').forEach(k => k.classList.toggle('active', k.dataset.view === active));
    document.querySelectorAll('.inset').forEach(k => k.classList.toggle('active', k.dataset.view === active));
    $('cut').innerHTML = Rn.cutSVG(cur, active);
    Rn.insets(cur, { inj: $('in-inj'), injT: $('in-inj-t'), grain: $('in-grain'), grainT: $('in-grain-t'), noz: $('in-noz'), nozT: $('in-noz-t'), fin: $('in-fin'), finT: $('in-fin-t') });
    $('curve').innerHTML = Rn.curveSVG(row);
    $('r-peak').textContent = `최대 ${fmt(cur.peak, 0)} N · 평균 ${fmt(Math.max(0, row.thrust_avg), 0)} N`;
    $('r-isp').textContent = fmt(Math.max(0, row.isp), 0); $('r-imp').textContent = fmt(Math.max(0, row.impulse) / 1000, 1); $('r-bt').textContent = fmt(row.burn_time, 1);
    $('r-conf').textContent = Math.round(cur.conf * 100) + '% · ±' + fmt(row.isp_std, 1) + ' s'; $('conf-bar').style.width = (cur.conf * 100) + '%';
    const mcol = A.margin < 0 ? 'nogo' : (A.margin < 1 || A.margin > 3.2) ? 'warn' : 'go';
    $('r-margin').textContent = (A.margin >= 0 ? '+' : '') + fmt(A.margin, 2) + ' cal'; $('r-margin').style.color = C[mcol];
    $('needle').style.left = mpos(A.margin) + '%';
    $('r-mass').textContent = fmt(A.m0, 1); $('r-tw').textContent = fmt(cur.tw, 1);
    const unpred = A.margin < 0 || !A.fit || cur.errs.length > 0;
    let apo = '???';
    if (!unpred) { const f = F.fly(cur, { wind: { spd: 0, dir: 0 } }); apo = f.outcome === 'APOGEE' ? fmt(f.apo, 0) : '???'; }
    $('r-apo').textContent = apo;
    $('m-margin').className = 'chip ' + mcol; $('m-margin').textContent = '안정 ' + fmt(A.margin, 1) + ' cal';
    $('m-len').textContent = '전장 ' + fmt(A.Ltot, 2) + ' m'; $('m-mass').textContent = fmt(A.m0, 1) + ' kg';
    $('m-apo').textContent = '예상 ' + apo + (apo === '???' ? '' : ' m');
    const list = [...cur.errs.map(c => ['nogo', c]), ...cur.warns.map(c => ['warn', c])];
    $('msgs').innerHTML = list.length ? list.map(([k, c]) => `<div class="status ${k}"><span class="dot"></span><span>${c}<small>${F.MSG[c] || ''}</small></span></div>`).join('')
      : `<div class="status go"><span class="dot"></span><span>ALL SYSTEMS GO<small>발사 준비 완료</small></span></div>`;
    $('btn-launch').disabled = !A.fit || !ME || ME.official_left <= 0;
    $('btn-test').disabled = !A.fit;
  }

  // ── views
  function view(v) {
    ['design', 'flight', 'board', 'me'].forEach(x => { $('v-' + x).hidden = x !== v; });
    document.querySelectorAll('nav.bottom button').forEach(b => { if (b.dataset.v === v) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current'); });
    $('actionbar').hidden = v !== 'design';
    if (v === 'flight') loadThree().then(() => { ensureSky(); if (sky) sky.resize(); }).catch(() => { });
    if (v === 'board') loadBoard();
    if (v === 'me') loadMe();
  }
  document.querySelectorAll('nav.bottom button').forEach(b => (b.onclick = () => view(b.dataset.v)));

  // 3D 라이브러리(600 KB)는 첫 화면을 막지 않도록 설계 화면이 뜬 뒤 받아 둔다
  let threeP = null;
  function loadThree() {
    if (window.THREE) return Promise.resolve();
    if (!threeP) threeP = new Promise((ok, no) => { const s = document.createElement('script'); s.src = '/static/vendor/three.min.js'; s.onload = ok; s.onerror = () => { threeP = null; no(new Error('3D 라이브러리를 받지 못했습니다')); }; document.head.appendChild(s); });
    return threeP;
  }
  function ensureSky() {
    if (sky || !$('nogl').hidden || !window.THREE) return;
    try { sky = Rn.Sky($('gl')); }
    catch (e) { $('nogl').hidden = false; }
  }

  // ── launch
  let pendingOfficial = false;
  $('btn-test').onclick = () => launch(false);
  $('btn-launch').onclick = () => {
    $('confirm-t').textContent = `공식 발사 1회를 사용합니다 (남은 ${ME.official_left}회). 현재 바람 ${META.wind.spd} m/s. 결과는 순위표에 반영됩니다.`;
    $('confirm').hidden = false;
  };
  $('confirm-no').onclick = () => ($('confirm').hidden = true);
  $('confirm-yes').onclick = () => { $('confirm').hidden = true; launch(true); };

  async function launch(official) {
    if (!cur || pendingOfficial) return;
    pendingOfficial = true;
    $('btn-launch').disabled = $('btn-test').disabled = true;
    try {
      const r = await api('/api/launch', { engine: engKey(), air: { body: st.body, fin: st.fin, nose: st.nose }, official });
      META.wind = r.wind; ME.official_left = r.official_left; chips();
      lastLaunch = { r, Dz: cur };
      view('flight'); play(lastLaunch);
    } catch (e) { toast(e.message); if (e.status === 401) showAuth(); }
    finally { pendingOfficial = false; render(); }
  }
  function outcomeInfo(fl) {
    if (fl.outcome === 'BURST') return { cls: 'nogo', t: 'BURST', s: '압력 한계 초과로 발사대에서 파열' };
    if (fl.outcome === 'CRASH') return { cls: 'nogo', t: fl.ev.tumble || fl.maxAoa > 0.8 ? 'CRASH · TUMBLE' : 'CRASH', s: `충돌 속도 ${fmt(fl.impactV, 0)} m/s · 최고 ${fmt(fl.apo, 0)} m` };
    if (fl.outcome === 'NO_LIFTOFF') return { cls: 'nogo', t: 'NO LIFTOFF', s: '추력이 무게보다 작아 이륙 실패' };
    if (fl.outcome === 'TIMEOUT') return { cls: 'warn', t: 'TIMEOUT', s: '비행 시간 초과' };
    if (fl.chug) return { cls: 'warn', t: 'ENGINE CHUG', s: `연소 불안정 · 정점 ${fmt(fl.apo, 0)} m` };
    if (fl.maxAoa > 0.25) return { cls: 'warn', t: 'WEATHERCOCK', s: `바람에 꺾임 · 정점 ${fmt(fl.apo, 0)} m` };
    return { cls: 'go', t: 'NOMINAL · APOGEE', s: `정점 ${fmt(fl.apo, 0)} m` };
  }
  function banner(cls, t, s) { const b = $('banner'); b.className = 'banner ' + cls; b.innerHTML = `${t}<small>${s}</small>`; }
  function hud(fl, ts, s) {
    const t = Math.max(0, ts);
    $('h-clock').textContent = (ts < 0 ? 'T−' : 'T+') + String(Math.floor(Math.abs(ts) / 60)).padStart(2, '0') + ':' + fmt(Math.abs(ts) % 60, 1).padStart(4, '0');
    $('h-alt').textContent = fmt(s.h, 0); $('h-spd').textContent = fmt(s.v * 3.6, 0);
    $('h-aoa').textContent = fmt(Rn.aoaAt(fl.tr, s.i), 0) + '°'; $('h-dr').textContent = fmt(Math.hypot(s.p[0], s.p[1]), 0);
    const ev = fl.ev, tEnd = fl.tr[fl.tr.length - 1].t;
    const times = { ign: fl.outcome === 'BURST' ? null : 0, rail: ev.rail, maxq: ev.maxq, bo: ev.bo, apo: ev.apo, land: tEnd };
    document.querySelectorAll('#evs div').forEach(d => {
      const e = d.dataset.e, tt = times[e]; d.classList.remove('on', 'bad');
      if (e === 'land') d.lastChild.textContent = fl.outcome === 'APOGEE' ? '착지' : '충돌';
      if (ts >= 0 && tt != null && t >= tt - 1e-6) d.classList.add(e === 'land' && fl.outcome !== 'APOGEE' ? 'bad' : 'on');
    });
  }
  function play(L) {
    const { r, Dz } = L, fl = r.fl;
    $('result').hidden = true;
    if (!window.THREE) { banner('idle', 'LOADING', '3D 화면 준비 중'); loadThree().then(() => play(L)).catch(() => { $('nogl').hidden = false; play(L); }); return; }
    ensureSky();
    const finish = () => { const last = fl.tr[fl.tr.length - 1]; hud(fl, last.t, last); showResult(L); };
    if (!sky) { finish(); return; }
    sky.load(Dz, fl, r.wind);
    banner('idle', 'T−3', '카운트다운');
    sky.start((ts, s) => {
      hud(fl, ts, s);
      if (ts < 0) banner('idle', 'T−' + Math.ceil(-ts * 2), '카운트다운');
      else if (s.chute) banner('go', 'UNDER CANOPY', '낙하산 하강');
      else if (fl.outcome !== 'APOGEE' && fl.maxAoa > 0.8 && ts > (fl.ev.rail || 0) + 0.4) banner('nogo', 'LOSS OF CONTROL', '자세 상실');
      else banner('idle', 'ASCENT', `T+${fmt(Math.max(0, ts), 1)} s`);
    }, finish);
  }
  function showResult(L) {
    const { r, Dz } = L, fl = r.fl, o = outcomeInfo(fl);
    banner(o.cls, o.t, o.s);
    $('res-status').className = 'status ' + o.cls; $('res-status-t').textContent = o.t;
    const ok = fl.outcome === 'APOGEE', target = META.target;
    $('res-apo').textContent = (ok ? fmt(fl.apo, 0) : '0') + ' m';
    $('res-delta').textContent = ok ? `목표 ${fmt(target, 0)} m 대비 ${fl.apo - target >= 0 ? '+' : ''}${fmt(fl.apo - target, 0)} m` : o.s;
    $('res-kind').textContent = r.official ? '공식 · 목표 점수' : '연습 · 순위 미반영';
    $('res-score').textContent = fmt(r.score.target, 0) + ' 점';
    $('res-rank').textContent = r.official && r.rank.target ? `목표 ${r.rank.target}위 · 최고고도 ${r.rank.max || '-'}위 / ${r.rank.of}명` : (r.official ? '순위 없음 (0점)' : '');
    $('res-mar').textContent = fmt(r.margin, 2) + ' cal'; $('res-v').textContent = fmt(fl.maxV * 3.6, 0) + ' km/h';
    $('res-wind').textContent = `${r.wind.spd} m/s · ${r.wind.dir}°`;
    let tip;
    if (fl.outcome === 'CRASH' && r.margin < 1) tip = '핀을 키우거나 지관을 늘려 CG를 CP보다 앞으로 보내세요.';
    else if (fl.outcome === 'CRASH') tip = '레일을 떠날 때 너무 느렸어요. 추력/중량을 올려 보세요.';
    else if (fl.outcome === 'NO_LIFTOFF') tip = '추력이 부족합니다. 압력이나 인젝터를 올리거나 산화제를 줄여 보세요.';
    else if (fl.chug) tip = '인젝터 차압이 부족합니다. 인젝터를 줄이거나 압력을 올리세요.';
    else if (fl.maxAoa > 0.25) tip = '바람에 꺾였어요. 핀을 조금 줄이거나 추력을 올려 보세요.';
    else if (fl.apo > target) tip = '목표 초과. 산화제를 줄이면 정확도와 절약 점수가 함께 오릅니다.';
    else tip = '조금 모자라요. 노즐 확장비나 노즈콘으로 효율을 올려 보세요.';
    $('res-tip').textContent = '힌트 · ' + tip;
    $('result').hidden = false;
  }
  $('res-replay').onclick = () => lastLaunch && play(lastLaunch);
  $('res-design').onclick = () => view('design');

  // ── board / me
  let bmode = 'target';
  $('b-target').onclick = () => { bmode = 'target'; loadBoard(); };
  $('b-max').onclick = () => { bmode = 'max'; loadBoard(); };
  async function loadBoard() {
    $('b-target').setAttribute('aria-selected', bmode === 'target'); $('b-max').setAttribute('aria-selected', bmode === 'max');
    $('b-desc').textContent = bmode === 'target' ? '목표 고도에 가까울수록, 산화제를 적게 쓸수록 높은 점수 (공식 발사 중 최고 기록)' : '추락하지 않고 가장 높이 올라간 기록 (공식 발사 중 최고 기록)';
    try {
      const [b, me] = await Promise.all([api('/api/board?mode=' + bmode), api('/api/me')]);
      ME = me; chips();
      const rk = me.rank;
      $('myrank').innerHTML = `<div><span class="lbl">목표 고도</span><b>${rk.target ? rk.target + '위' : '—'}</b><span class="mono" style="font-size:11px;color:var(--muted)">${rk.target_score ? fmt(rk.target_score) + '점' : '공식 기록 없음'}</span></div><div><span class="lbl">최고 고도</span><b>${rk.max ? rk.max + '위' : '—'}</b><span class="mono" style="font-size:11px;color:var(--muted)">${rk.max_score ? fmt(rk.max_score) + ' m' : '공식 기록 없음'}</span></div>`;
      let rows = b.rows.map(r => ({ ...r, me: false }));
      const ai = b.ai && b.ai[bmode];
      const html = rows.map(r => `<div class="lb-row ${r.rank <= 3 ? 'top' : ''}"><span class="r">${String(r.rank).padStart(2, '0')}</span><span class="n"><b>${esc(r.name)}</b><small>${esc(r.dept)}</small></span><span class="h">${fmt(r.apo)} m</span><span class="s">${bmode === 'target' ? fmt(r.score) : fmt(r.score) + 'm'}</span></div>`);
      if (ai) html.splice(Math.min(html.length, rows.filter(r => r.score > ai.score).length), 0, `<div class="lb-row ai"><span class="r">AI</span><span class="n"><b>TAS-1 MDO</b><small>최적화기 기준선 · ${esc(ai.note || '')}</small></span><span class="h">${fmt(ai.apo)} m</span><span class="s">${bmode === 'target' ? fmt(ai.score) : fmt(ai.score) + 'm'}</span></div>`);
      $('lb').innerHTML = html.join('') || '<div class="empty">아직 공식 기록이 없습니다. 첫 번째 주인공이 되어 보세요.</div>';
    } catch (e) { toast(e.message); }
  }
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  async function loadMe() {
    try {
      ME = await api('/api/me'); chips();
      $('me-name').textContent = ME.name;
      $('me-rows').innerHTML = [['학번', ME.student_id], ['학과', ME.dept], ['전화번호', ME.phone], ['공식 발사', `${META.official_limit - ME.official_left} / ${META.official_limit} 사용`]].map(([a, b]) => `<div><span>${a}</span><b>${esc(b)}</b></div>`).join('');
      $('me-count').textContent = ME.history.length + '회';
      $('hist').innerHTML = ME.history.map(h => `<div><span class="o" style="color:${h.outcome === 'APOGEE' ? C.go : C.nogo}">${h.official ? '공식' : '연습'}</span><span>${new Date(h.created_at * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })} · ${h.outcome === 'APOGEE' ? fmt(h.apo) + ' m' : h.outcome}</span><span class="mono">${h.official ? fmt(h.score_target) + '점' : ''}</span></div>`).join('') || '<div class="empty">아직 발사 기록이 없습니다.</div>';
    } catch (e) { toast(e.message); }
  }
  function chips() {
    if (META) { const w = META.wind; $('wind-chip').textContent = `바람 ${w.spd} m/s`; }
    if (ME) {
      $('left-chip').textContent = `공식 ${ME.official_left}/${META.official_limit}`;
      $('left-chip').className = 'chip ' + (ME.official_left ? '' : 'warn');
      $('btn-launch-s').textContent = ME.official_left ? `${ME.official_left}회 남음` : '모두 사용';
    }
  }

  async function enter() {
    ME = await api('/api/me');
    $('auth').hidden = true; $('app').hidden = false;
    chips(); if (!$('k-eng').children.length) buildKnobs();
    view('design'); update();
    setTimeout(() => loadThree().catch(() => { }), 1200);
  }
  async function boot() {
    try { META = await api('/api/meta'); } catch (e) { document.body.innerHTML = '<div class="empty">서버에 연결할 수 없습니다. 잠시 후 새로고침하세요.</div>'; return; }
    $('b-target-v').textContent = fmt(META.target);
    $('lut-info').textContent = `${fmt(META.lut.n)}개 설계 · ${META.lut.device.toUpperCase()} ${META.lut.inference_s}초 계산`;
    setInterval(async () => { try { META.wind = (await api('/api/meta')).wind; chips(); } catch (_) { } }, 120000);
    try { await enter(); } catch (e) { if (e.status === 401) showAuth(); else toast(e.message); }
  }
  boot();
})();
