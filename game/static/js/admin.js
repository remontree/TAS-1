(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const fmt = (x, d = 0) => x == null ? '—' : Number(x).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const tm = t => t ? new Date(t * 1000).toLocaleString('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—';
  async function api(p, body) {
    const o = { credentials: 'same-origin', headers: {} };
    if (body !== undefined) { o.method = 'POST'; o.headers['Content-Type'] = 'application/json'; o.headers['X-TAS1'] = '1'; o.body = JSON.stringify(body); }
    const r = await fetch(p, o); let j = null; try { j = await r.json(); } catch (_) { }
    if (!r.ok) { const e = new Error((j && j.error) || '오류'); e.status = r.status; throw e; } return j;
  }
  let T; const toast = m => { const t = $('toast'); t.textContent = m; t.hidden = false; clearTimeout(T); T = setTimeout(() => (t.hidden = true), 3000); };
  let players = [];
  $('login').onsubmit = async e => { e.preventDefault(); $('lerr').textContent = ''; try { await api('/api/admin/login', { password: $('pw').value }); $('pw').value = ''; start(); } catch (er) { $('lerr').textContent = er.message; } };
  $('out').onclick = async () => { await api('/api/admin/logout', {}); location.reload(); };
  $('q').oninput = draw;
  function draw() {
    const q = $('q').value.trim();
    $('players').innerHTML = players.filter(p => !q || (p.name + p.student_id + p.dept).includes(q)).map(p => `<tr class="${p.hidden ? 'hid' : ''}"><td class="n">${p.id}</td><td>${esc(p.student_id)}</td><td>${esc(p.name)}</td><td>${esc(p.dept)}</td><td>${esc(p.phone)}</td><td class="n">${p.off}</td><td class="n">${p.total}</td><td class="n">${fmt(p.best_t)}</td><td class="n">${fmt(p.best_m)}</td><td>${tm(p.created_at)}</td><td>${tm(p.last_seen)}</td><td><button class="act" data-a="hide" data-id="${p.id}">${p.hidden ? '보이기' : '숨기기'}</button><button class="act" data-a="reset_official" data-id="${p.id}">공식 초기화</button><button class="act" data-a="logout" data-id="${p.id}">강제 로그아웃</button></td></tr>`).join('');
  }
  $('players').onclick = async e => {
    const b = e.target.closest('button.act'); if (!b) return;
    if (b.dataset.armed !== '1') { b.dataset.armed = '1'; const o = b.textContent; b.textContent = '한 번 더'; setTimeout(() => { b.dataset.armed = ''; b.textContent = o; }, 2500); return; }
    try { await api('/api/admin/player/' + b.dataset.id, { action: b.dataset.a }); toast('적용했습니다'); load(); } catch (er) { toast(er.message); }
  };
  async function load() {
    try {
      const [s, p] = await Promise.all([api('/api/admin/stats'), api('/api/admin/players')]);
      players = p; draw();
      const st = s.stats || {};
      const up = s.uptime, h = Math.floor(up / 3600), m = Math.floor(up / 60) % 60;
      $('cards').innerHTML = [
        ['참가자', fmt(p.length), `활성 공식 ${p.filter(x => x.off).length}명`],
        ['발사', fmt(st.launches || 0), '이번 프로세스 기준'],
        ['요청/분', fmt(s.req_per_min), `총 ${fmt(st.requests || 0)}`],
        ['가동 시간', `${h}h ${m}m`, '재시작 시 초기화'],
        ['비행 계산', s.node_ms ? fmt(s.node_ms, 0) + ' ms' : '—', 'node 재계산 평균'],
        ['로그인 실패', fmt(st.login_fail || 0), `관리자 실패 ${fmt(st.admin_fail || 0)}`],
        ['차단', fmt((s.bans || []).length), `누적 ${fmt(st.bans || 0)} · 404 ${fmt(st['404'] || 0)}`],
        ['바람', `${s.wind.spd} m/s`, `${s.wind.dir}° · 매 정시 변경`],
      ].map(([a, b, c]) => `<div><span class="lbl">${a}</span><b>${b}</b><small>${c}</small></div>`).join('');
      $('bans').innerHTML = (s.bans || []).map(b => `<div class="mono" style="font-size:12.5px;display:flex;gap:10px;align-items:center;padding:4px 0">${esc(b.ip)} · ${esc(b.reason)} · ${tm(b.until)}까지 <button class="act" data-ip="${esc(b.ip)}">해제</button></div>`).join('') || '<span style="color:var(--muted);font-size:13px">없음</span>';
      const L = s.lut;
      $('lut').textContent = `모델 ${L.model} · ${L.device} · ${L.n}개 설계 (${L.n_ok}개 제약 통과)\n계산 ${L.inference_s}초 · 생성 ${L.built_at}\n파이프 ${L.pipe} · 온도 ${L.temperature_K} K · 목표 ${s.target} m`;
    } catch (er) { if (er.status === 401) { $('dash').hidden = true; $('login').hidden = false; $('out').hidden = true; } else toast(er.message); }
  }
  $('bans').onclick = async e => { const b = e.target.closest('button[data-ip]'); if (!b) return; await api('/api/admin/unban', { ip: b.dataset.ip }); load(); };
  let iv;
  function start() { $('login').hidden = true; $('dash').hidden = false; $('out').hidden = false; load(); clearInterval(iv); iv = setInterval(load, 5000); }
  api('/api/admin/stats').then(start).catch(() => { });
})();
