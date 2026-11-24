/* 단면도 SVG · 확대도 · 2D 궤적 · 3D 비행 장면 (앱과 대형 화면 공용) */
(function () {
  'use strict';
  const F = window.Flight;
  // IBM Carbon g100 토큰
  const C = { fg: '#f4f4f4', soft: '#c6c6c6', muted: '#a8a8a8', dim: '#6f6f6f', line: '#393939', strong: '#6f6f6f', panel: '#393939', go: '#42be65', warn: '#f1c21b', nogo: '#fa4d56', info: '#78a9ff', blue: '#4589ff', flame: '#ff832b' };
  const fmt = (x, d = 0) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  let uid = 0;

  function cutSVG(Dz, act, opts) {
    opts = opts || {};
    const A = Dz.A, g = Dz.row.geom;
    const W = 1000, H = opts.compact ? 200 : 236, cy = H / 2 + (opts.compact ? 0 : 4);
    const sx = (W - 30) / Math.max(3.4, A.xEnd + 0.1), sy = sx * 2;  // 현재 기체 길이에 맞춰 화면을 채운다
    const X = x => 14 + x * sx, Y = r => cy - r * sy, Yb = r => cy + r * sy;
    const hi = v => (act === v ? C.info : C.fg), sw = v => (act === v ? 2.4 : 1.2);
    const bad = !A.fit, ec = v => (bad ? C.nogo : hi(v));
    const hid = 'h' + (++uid);
    let s = `<defs><pattern id="${hid}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="#262626"/><line x1="0" y1="0" x2="0" y2="7" stroke="#6f6f6f" stroke-width="2.4"/></pattern></defs>`;
    s += `<line x1="${X(0) - 6}" y1="${cy}" x2="${X(A.xEnd) + 12}" y2="${cy}" stroke="${C.line}" stroke-dasharray="4 4"/>`;
    const xb = A.Ltot - F.CR;
    const fin = k => `<path d="M${X(xb)} ${cy - k * F.R * sy} L${X(xb + F.XSW)} ${cy - k * (F.R + A.s) * sy} L${X(xb + F.XSW + F.CT)} ${cy - k * (F.R + A.s) * sy} L${X(A.Ltot)} ${cy - k * F.R * sy} Z" fill="${C.panel}" stroke="${hi('fin')}" stroke-width="${sw('fin')}"/>`;
    s += fin(1) + fin(-1);
    s += `<rect x="${X(F.LN)}" y="${Y(F.R)}" width="${A.Lb * sx}" height="${2 * F.R * sy}" fill="#262626" stroke="${hi('body')}" stroke-width="${sw('body')}"/>`;
    let np = '';
    for (let j = 0; j <= 30; j++) { const x = j / 30 * F.LN; np += (j ? ' L' : 'M') + X(x).toFixed(1) + ' ' + Y(F.noseR(A.nose.id, x)).toFixed(1); }
    for (let j = 30; j >= 0; j--) { const x = j / 30 * F.LN; np += ' L' + X(x).toFixed(1) + ' ' + Yb(F.noseR(A.nose.id, x)).toFixed(1); }
    s += `<path d="${np} Z" fill="#262626" stroke="${hi('nose')}" stroke-width="${sw('nose')}"/>`;
    s += `<rect x="${X(F.LN + 0.03)}" y="${Y(F.R * 0.78)}" width="${(F.BAY - 0.06) * sx}" height="${1.56 * F.R * sy}" fill="none" stroke="${C.dim}" stroke-dasharray="3 3"/>`;
    const tr = 0.059;
    s += `<rect x="${X(A.xT0)}" y="${Y(tr)}" width="${A.Lt * sx}" height="${2 * tr * sy}" rx="${tr * sy * 0.6}" fill="#1c1c1c" stroke="${ec('tank')}" stroke-width="${sw('tank')}"/>`;
    const liq = A.Lt * 0.85;
    s += `<rect x="${X(A.xT0 + A.Lt - liq)}" y="${Y(tr * 0.84)}" width="${Math.max(0, liq * sx - 4)}" height="${2 * tr * 0.84 * sy}" rx="${tr * sy * 0.45}" fill="rgba(69,137,255,.28)"/>`;
    s += `<rect x="${X(A.xI0)}" y="${Y(tr)}" width="${F.L_INJ * sx}" height="${2 * tr * sy}" fill="${act === 'inj' ? 'rgba(69,137,255,.45)' : '#525252'}" stroke="${ec('inj')}" stroke-width="${sw('inj')}"/>`;
    const nh = Math.min(g.inj_n, 24);
    for (let j = 0; j < nh; j++) { const yy = Y(tr * 0.85) + (j + 0.5) / nh * (1.7 * tr * sy); s += `<line x1="${X(A.xI0 + F.L_INJ)}" y1="${yy}" x2="${X(A.xI0 + F.L_INJ) + 7}" y2="${yy}" stroke="${C.info}" stroke-width="1.2" opacity=".85"/>`; }
    s += `<rect x="${X(A.xC0)}" y="${Y(tr)}" width="${A.Lc * sx}" height="${2 * tr * sy}" fill="#1c1c1c" stroke="${ec('grain')}" stroke-width="${sw('grain')}"/>`;
    const g0 = A.xC0 + (A.Lc - g.fuel_len) / 2;
    s += `<rect x="${X(g0)}" y="${Y(g.outer_r)}" width="${g.fuel_len * sx}" height="${2 * g.outer_r * sy}" fill="url(#${hid})" stroke="${act === 'grain' ? C.info : 'none'}"/>`;
    s += `<rect x="${X(g0)}" y="${cy - g.port_r * sy}" width="${g.fuel_len * sx}" height="${2 * g.port_r * sy}" fill="#161616"/>`;
    const nz = A.xNz0, rt = g.throat_r, re = g.exit_r;
    s += `<path d="M${X(nz)} ${Y(0.05)} L${X(nz + 0.03)} ${Y(rt)} L${X(nz + A.Lnz)} ${Y(re)} L${X(nz + A.Lnz)} ${Yb(re)} L${X(nz + 0.03)} ${Yb(rt)} L${X(nz)} ${Yb(0.05)} Z" fill="#525252" stroke="${ec('nozzle')}" stroke-width="${sw('nozzle')}"/>`;
    const fsz = opts.compact ? 24 : 15;
    if (bad) {
      s += `<rect x="${X(A.Ltot)}" y="${Y(F.R) - 4}" width="${A.over * sx}" height="${2 * F.R * sy + 8}" fill="rgba(255,77,77,.1)" stroke="${C.nogo}" stroke-dasharray="5 4"/>`;
      s += `<text x="${X(A.Ltot + A.over / 2)}" y="${Yb(F.R) + fsz + 6}" text-anchor="middle" fill="${C.nogo}" font-family="IBM Plex Mono, monospace" font-size="${fsz}">+${fmt(A.over, 2)} m 초과</text>`;
    }
    const cgx = X(A.cg0), cpx = X(A.cp), cr = opts.compact ? 13 : 11;
    s += `<circle cx="${cpx}" cy="${cy}" r="${cr * 0.72}" fill="${C.nogo}"/>`;
    s += `<circle cx="${cgx}" cy="${cy}" r="${cr}" fill="#161616" stroke="${C.fg}" stroke-width="1.8"/><path d="M${cgx} ${cy - cr} A${cr} ${cr} 0 0 1 ${cgx + cr} ${cy} L${cgx} ${cy} Z M${cgx} ${cy + cr} A${cr} ${cr} 0 0 1 ${cgx - cr} ${cy} L${cgx} ${cy} Z" fill="${C.fg}"/>`;
    const mcol = A.margin < 0 ? C.nogo : (A.margin < 1 || A.margin > 3.2) ? C.warn : C.go;
    if (!opts.compact) {
      const by = Yb(F.R) + 20;
      s += `<line x1="${cgx}" y1="${by}" x2="${cpx}" y2="${by}" stroke="${mcol}" stroke-width="3"/>`;
      s += `<line x1="${cgx}" y1="${by - 6}" x2="${cgx}" y2="${by + 6}" stroke="${C.fg}" stroke-width="1.4"/><line x1="${cpx}" y1="${by - 6}" x2="${cpx}" y2="${by + 6}" stroke="${C.nogo}" stroke-width="1.4"/>`;
      s += `<text x="${(cgx + cpx) / 2}" y="${by + 22}" text-anchor="middle" fill="${mcol}" font-family="IBM Plex Mono, monospace" font-size="16">${A.margin >= 0 ? '+' : ''}${fmt(A.margin, 2)} cal</text>`;
      s += `<text x="${cgx}" y="${Y(F.R) - 9}" text-anchor="middle" fill="${C.fg}" font-family="IBM Plex Mono, monospace" font-size="14">CG</text>`;
      s += `<text x="${cpx}" y="${Y(F.R) - 9}" text-anchor="middle" fill="${C.nogo}" font-family="IBM Plex Mono, monospace" font-size="14">CP</text>`;
      s += `<text x="${X(0)}" y="${H - 4}" fill="${C.dim}" font-family="IBM Plex Mono, monospace" font-size="14">L ${fmt(A.Ltot, 2)} m · ⌀130 mm · 세로 2배</text>`;
    }
    return s;
  }

  function insets(Dz, el) {
    const g = Dz.row.geom;
    let s = `<circle cx="60" cy="46" r="40" fill="#1c1c1c" stroke="${C.fg}"/>`;
    let left = g.inj_n, ring = 0; const caps = [1, 6, 12, 18, 24, 30];
    while (left > 0 && ring < caps.length) { const c = Math.min(left, caps[ring]); const rr = ring * 7; for (let j = 0; j < c; j++) { const a = j / c * Math.PI * 2 + ring * 0.3; s += `<circle cx="${(60 + rr * Math.cos(a)).toFixed(1)}" cy="${(46 + rr * Math.sin(a)).toFixed(1)}" r="2" fill="${C.info}"/>`; } left -= c; ring++; }
    el.inj.innerHTML = s; el.injT.textContent = `${g.inj_n}구 · ⌀${fmt(g.inj_r * 2000, 2)}`;
    const k = 38 / 0.055, hid = 'g' + (++uid);
    s = `<defs><pattern id="${hid}" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#262626"/><line x1="0" y1="0" x2="0" y2="5" stroke="#6f6f6f" stroke-width="2"/></pattern></defs>`;
    s += `<circle cx="44" cy="46" r="38" fill="none" stroke="${C.strong}"/><circle cx="44" cy="46" r="${g.outer_r * k}" fill="url(#${hid})" stroke="${C.fg}"/>`;
    for (let j = 0; j < g.port_n; j++) { const a = j / g.port_n * Math.PI * 2; s += `<circle cx="${(44 + g.port_c * k * Math.cos(a)).toFixed(1)}" cy="${(46 + g.port_c * k * Math.sin(a)).toFixed(1)}" r="${(g.port_r * k).toFixed(1)}" fill="#161616" stroke="${C.muted}"/>`; }
    const gl = Math.max(4, g.fuel_len / 1.0 * 80);
    s += `<rect x="96" y="${86 - gl}" width="14" height="${gl}" fill="url(#${hid})" stroke="${C.fg}"/>`;
    el.grain.innerHTML = s; el.grainT.textContent = `L ${fmt(g.fuel_len, 2)} m · 포트 ${g.port_n}`;
    const eps = (g.exit_r / g.throat_r) ** 2, kk = 300;
    s = `<path d="M6 ${46 - 34} L40 ${46 - g.throat_r * kk} L${40 + 18 + Math.min(eps, 16) * 3.8} ${46 - g.exit_r * kk} L${40 + 18 + Math.min(eps, 16) * 3.8} ${46 + g.exit_r * kk} L40 ${46 + g.throat_r * kk} L6 ${46 + 34} Z" fill="#525252" stroke="${C.fg}"/>`;
    el.noz.innerHTML = s; el.nozT.textContent = `ε ${fmt(eps, 1)} · Isp ${fmt(Math.max(0, Dz.row.isp), 0)}`;
    const kf = 260, sp = Dz.A.s * kf, rx = 6, ry = 86;
    s = `<line x1="2" y1="${ry}" x2="118" y2="${ry}" stroke="${C.strong}"/><path d="M${rx} ${ry} L${rx + F.XSW * kf} ${ry - sp} L${rx + (F.XSW + F.CT) * kf} ${ry - sp} L${rx + F.CR * kf} ${ry} Z" fill="${C.panel}" stroke="${C.fg}"/>`;
    el.fin.innerHTML = s; el.finT.textContent = `${fmt(Dz.A.s * 1000, 0)} mm ×4`;
  }

  function curveSVG(row, opts) {
    opts = opts || {};
    const th = row.thrust, dt = row.dt;
    const tmax = Math.max(row.t_end * 1.15, 3), fmaxS = opts.fmax || 3000;
    const pts = th.map((f, i) => `${(i * dt / tmax * 298 + 1).toFixed(1)},${(62 - Math.min(f, fmaxS) / fmaxS * 58).toFixed(1)}`);
    const bad = (row.codes || []).length > 0;
    return `<line x1="0" y1="62" x2="300" y2="62" stroke="${C.line}"/><polygon points="1,62 ${pts.join(' ')} ${pts.length ? pts[pts.length - 1].split(',')[0] : 1},62" fill="rgba(69,137,255,.16)"/><polyline points="${pts.join(' ')}" fill="none" stroke="${bad ? C.nogo : C.blue}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`;
  }

  function sizeCanvas(c) {
    const r = c.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.max(10, r.width * dpr); c.height = Math.max(10, r.height * dpr);
    const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); return { x, w: r.width, h: r.height };
  }
  // paths: [{pts:[[d,h]..], bad, alpha, lw}]
  function drawSide(c, paths, opt) {
    opt = opt || {};
    const { x, w, h } = sizeCanvas(c); x.clearRect(0, 0, w, h);
    const all = paths.flatMap(p => p.pts);
    const hmax = Math.max(1000, Math.ceil(Math.max(opt.target || 0, ...all.map(p => p[1])) * 1.1 / 1000) * 1000);
    const dmax = Math.max(500, ...all.map(p => p[0])) * 1.08;
    const pl = 34, pb = 16, X = d => pl + d / dmax * (w - pl - 8), Y = a => h - pb - a / hmax * (h - pb - 8);
    x.font = '10px "IBM Plex Mono", monospace'; x.lineWidth = 1;
    for (let a = 0; a <= hmax; a += hmax / 4) { x.strokeStyle = C.line; x.beginPath(); x.moveTo(pl, Y(a)); x.lineTo(w, Y(a)); x.stroke(); x.fillStyle = C.dim; x.fillText(a >= 1000 ? (a / 1000) + 'k' : a, 2, Y(a) + 3); }
    x.textAlign = 'right'; x.fillText('수평 ' + (dmax >= 1100 ? fmt(dmax / 1080, 1) + ' km' : fmt(dmax / 1.08, 0) + ' m') + ' →', w - 4, h - 3); x.textAlign = 'left';
    if (opt.target) { x.strokeStyle = 'rgba(120,169,255,.8)'; x.setLineDash([4, 4]); x.beginPath(); x.moveTo(pl, Y(opt.target)); x.lineTo(w, Y(opt.target)); x.stroke(); x.setLineDash([]); }
    paths.forEach(p => {
      if (!p.pts.length) return;
      x.globalAlpha = p.alpha || 1; x.strokeStyle = p.bad ? C.nogo : C.fg; x.lineWidth = p.lw || 1.6; x.beginPath();
      p.pts.forEach((q, i) => (i ? x.lineTo(X(q[0]), Y(q[1])) : x.moveTo(X(q[0]), Y(q[1])))); x.stroke(); x.globalAlpha = 1;
      const l = p.pts[p.pts.length - 1];
      if (p.bad && p.done) { x.strokeStyle = C.nogo; x.lineWidth = 2; x.beginPath(); x.moveTo(X(l[0]) - 5, Y(l[1]) - 5); x.lineTo(X(l[0]) + 5, Y(l[1]) + 5); x.moveTo(X(l[0]) + 5, Y(l[1]) - 5); x.lineTo(X(l[0]) - 5, Y(l[1]) + 5); x.stroke(); }
      else { x.fillStyle = p.bad ? C.nogo : C.fg; x.beginPath(); x.arc(X(l[0]), Y(l[1]), 3, 0, 7); x.fill(); }
    });
  }

  function sample(tr, ts) {
    if (!tr.length) return { p: [0, 0, 0], u: [0, 0, 1], v: 0, h: 0, i: 0 };
    let lo = 0, hi = tr.length - 1;
    if (ts <= tr[0].t) return Object.assign({}, tr[0], { i: 0 });
    if (ts >= tr[hi].t) return Object.assign({}, tr[hi], { i: hi });
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tr[m].t <= ts) lo = m; else hi = m; }
    const a = tr[lo], b = tr[hi], k = (ts - a.t) / (b.t - a.t || 1), L = (p, q) => p + (q - p) * k;
    return { p: [L(a.p[0], b.p[0]), L(a.p[1], b.p[1]), L(a.p[2], b.p[2])], u: [L(a.u[0], b.u[0]), L(a.u[1], b.u[1]), L(a.u[2], b.u[2])], v: L(a.v, b.v), h: L(a.h, b.h), chute: b.chute, i: lo };
  }
  function aoaAt(tr, i) {
    if (!(i >= 1 && i < tr.length)) return 0;
    const a = tr[i - 1], b = tr[i];
    const d = [b.p[0] - a.p[0], b.p[1] - a.p[1], b.p[2] - a.p[2]], l = Math.hypot(d[0], d[1], d[2]);
    if (l < 1e-3 || b.chute) return 0;
    const u = b.u, ul = Math.hypot(u[0], u[1], u[2]) || 1;
    return Math.acos(Math.max(-1, Math.min(1, (d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) / (l * ul)))) * 180 / Math.PI;
  }

  // ── 3D
  function Sky(canvas, opts) {
    opts = opts || {};
    const T = window.THREE;
    if (!T) throw new Error('three.js missing');
    const mobile = matchMedia('(max-width: 760px)').matches;
    const renderer = new T.WebGLRenderer({ canvas, antialias: !mobile, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2));
    const scene = new T.Scene(); scene.background = new T.Color(0x161616); scene.fog = new T.Fog(0x161616, 4000, 20000);
    const cam = new T.PerspectiveCamera(50, 1, 0.3, 60000);
    scene.add(new T.HemisphereLight(0xffffff, 0x262626, 0.9));
    const dl = new T.DirectionalLight(0xffffff, 0.8); dl.position.set(300, 800, 200); scene.add(dl);
    scene.add(new T.GridHelper(24000, 96, 0x393939, 0x1f1f1f));
    const g2 = new T.GridHelper(400, 40, 0x525252, 0x262626); g2.position.y = 0.05; scene.add(g2);
    [1000, 2000, 4000, 8000].forEach(r => { const pts = new T.EllipseCurve(0, 0, r, r, 0, Math.PI * 2).getPoints(128).map(p => new T.Vector3(p.x, 0.1, p.y)); scene.add(new T.Line(new T.BufferGeometry().setFromPoints(pts), new T.LineBasicMaterial({ color: 0x262626 }))); });
    const toV = p => new T.Vector3(p[0], p[2], -p[1]);
    scene.add(new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(0, 0, 0), toV([0, Math.cos(87 * Math.PI / 180) * 8, Math.sin(87 * Math.PI / 180) * 8])]), new T.LineBasicMaterial({ color: 0x8d8d8d })));
    scene.add(new T.Mesh(new T.CylinderGeometry(3, 3, 0.2, 24), new T.MeshStandardMaterial({ color: 0x393939 })));
    let arrow = null;
    function setWind(w) {
      if (arrow) scene.remove(arrow);
      if (!w || !w.spd) return;
      const wr = w.dir * Math.PI / 180, dir = toV([-Math.sin(wr), -Math.cos(wr), 0]).normalize();
      arrow = new T.ArrowHelper(dir, new T.Vector3(0, 2, 0).sub(dir.clone().multiplyScalar(30)), 20 + w.spd * 10, 0x78a9ff, 12, 6); scene.add(arrow);
    }
    const MAXP = 8000;
    const mk = col => { const geo = new T.BufferGeometry(); geo.setAttribute('position', new T.BufferAttribute(new Float32Array(MAXP * 3), 3)); const l = new T.Line(geo, new T.LineBasicMaterial({ color: col })); l.frustumCulled = false; scene.add(l); return l; };
    const trailA = mk(0xf4f4f4), trailC = mk(0x78a9ff);
    const drop = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3()]), new T.LineDashedMaterial({ color: 0x6f6f6f, dashSize: 8, gapSize: 6 })); drop.frustumCulled = false; scene.add(drop);
    const landRing = new T.Mesh(new T.RingGeometry(14, 18, 40), new T.MeshBasicMaterial({ color: 0x78a9ff, side: T.DoubleSide })); landRing.rotation.x = -Math.PI / 2; landRing.visible = false; scene.add(landRing);
    const scorch = new T.Mesh(new T.CircleGeometry(16, 32), new T.MeshBasicMaterial({ color: 0x3a1a10, side: T.DoubleSide })); scorch.rotation.x = -Math.PI / 2; scorch.visible = false; scene.add(scorch);
    const NP = 220, pg = new T.BufferGeometry(); pg.setAttribute('position', new T.BufferAttribute(new Float32Array(NP * 3), 3));
    const pm = new T.PointsMaterial({ color: 0xff832b, size: 4, transparent: true, opacity: 1 });
    const parts = new T.Points(pg, pm); parts.visible = false; parts.frustumCulled = false; scene.add(parts);
    const pvel = [];
    let rocket = null, flame = null, canopy = null, Dz = null, fl = null, play = null, dirty = true, visible = true;
    function build() {
      if (rocket) scene.remove(rocket);
      const A = Dz.A, cg = A.cg0; rocket = new T.Group();
      const mB = new T.MeshStandardMaterial({ color: 0xe0e0e0, roughness: .55, metalness: .15 });
      const mD = new T.MeshStandardMaterial({ color: 0x393939, roughness: .6, metalness: .3, side: T.DoubleSide });
      const body = new T.Mesh(new T.CylinderGeometry(F.R, F.R, A.Lb, 24), mB); body.position.y = cg - (F.LN + A.Lb / 2); rocket.add(body);
      const pts = []; for (let j = 0; j <= 24; j++) { const x = j / 24 * F.LN; pts.push(new T.Vector2(Math.max(0.0005, F.noseR(A.nose.id, x)), cg - x)); }
      rocket.add(new T.Mesh(new T.LatheGeometry(pts, 24), mB));
      const xb = A.Ltot - F.CR;
      for (let k = 0; k < 4; k++) {
        const sh = new T.Shape(); sh.moveTo(F.R, cg - xb); sh.lineTo(F.R + A.s, cg - (xb + F.XSW)); sh.lineTo(F.R + A.s, cg - (xb + F.XSW + F.CT)); sh.lineTo(F.R, cg - A.Ltot); sh.lineTo(F.R, cg - xb);
        const f = new T.Mesh(new T.ShapeGeometry(sh), mD); f.rotation.y = k * Math.PI / 2 + Math.PI / 4; rocket.add(f);
      }
      const re = Dz.row.geom.exit_r;
      const nz = new T.Mesh(new T.CylinderGeometry(0.03, re, A.Lnz, 16, 1, true), mD); nz.position.y = cg - A.Ltot - A.Lnz / 2 + 0.02; rocket.add(nz);
      flame = new T.Mesh(new T.ConeGeometry(re * 1.3, 1.0, 14), new T.MeshBasicMaterial({ color: 0xff832b, transparent: true, opacity: .9 }));
      flame.rotation.x = Math.PI; flame.position.y = cg - A.Ltot - A.Lnz - 0.45; rocket.add(flame);
      canopy = new T.Mesh(new T.SphereGeometry(0.6, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), new T.MeshBasicMaterial({ color: 0x78a9ff, side: T.DoubleSide, transparent: true, opacity: .85 }));
      canopy.position.y = cg + 2.4; canopy.visible = false; rocket.add(canopy);
      scene.add(rocket);
    }
    const camS = { az: -0.9, el: 0.32, mul: 1, target: new T.Vector3(0, 20, 0) };
    let drag = null;
    canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY }; try { canvas.setPointerCapture(e.pointerId); } catch (_) { } });
    canvas.addEventListener('pointermove', e => { if (!drag) return; camS.az -= (e.clientX - drag.x) * 0.006; camS.el = Math.max(0.04, Math.min(1.45, camS.el + (e.clientY - drag.y) * 0.005)); drag = { x: e.clientX, y: e.clientY }; dirty = true; });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('pointercancel', () => { drag = null; });
    canvas.addEventListener('wheel', e => { e.preventDefault(); camS.mul = Math.max(0.25, Math.min(5, camS.mul * Math.exp(e.deltaY * 0.001))); dirty = true; }, { passive: false });
    function resize() { const r = canvas.parentElement.getBoundingClientRect(); if (r.width < 2) return; renderer.setSize(r.width, r.height, false); cam.aspect = r.width / Math.max(1, r.height); cam.updateProjectionMatrix(); dirty = true; }
    new ResizeObserver(resize).observe(canvas.parentElement); resize();
    new IntersectionObserver(es => { visible = es[0].isIntersecting; dirty = true; }).observe(canvas);
    function setTrail(upto) {
      const pa = trailA.geometry.attributes.position, pc = trailC.geometry.attributes.position; let na = 0, nc = 0, lastA = null;
      for (let i = 0; i < fl.tr.length && na + nc < MAXP - 2; i++) {
        const r = fl.tr[i]; if (r.t > upto) break; const v = toV(r.p);
        if (!r.chute) { pa.setXYZ(na++, v.x, v.y, v.z); lastA = v; }
        else { if (nc === 0 && lastA) pc.setXYZ(nc++, lastA.x, lastA.y, lastA.z); pc.setXYZ(nc++, v.x, v.y, v.z); }
      }
      trailA.geometry.setDrawRange(0, na); trailC.geometry.setDrawRange(0, nc); pa.needsUpdate = true; pc.needsUpdate = true;
    }
    function boom(at, big) {
      parts.visible = true; pm.opacity = 1; pvel.length = 0; const pa = pg.attributes.position;
      for (let i = 0; i < NP; i++) { pa.setXYZ(i, at.x, at.y + 1, at.z); const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI * 0.5, sp = (big ? 32 : 22) * (0.3 + Math.random()); pvel.push([Math.cos(th) * Math.cos(ph) * sp, Math.sin(ph) * sp, Math.sin(th) * Math.cos(ph) * sp]); }
      pa.needsUpdate = true; scorch.position.set(at.x, 0.2, at.z); scorch.visible = true;
    }
    const burnEnd = () => (fl.chug ? Math.min(1.5, Dz.row.t_end) : Dz.row.t_end);
    function place(st, ts) {
      if (!rocket) return;
      const pos = toV(st.p); rocket.position.copy(pos);
      rocket.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), st.chute ? new T.Vector3(0, 1, 0) : toV(st.u).normalize());
      const burning = ts >= 0 && ts < burnEnd() && fl.outcome !== 'BURST';
      flame.visible = burning; if (burning) flame.scale.set(1, 0.8 + Math.random() * 0.5, 1);
      canopy.visible = !!st.chute;
      const pd = drop.geometry.attributes.position; pd.setXYZ(0, pos.x, pos.y, pos.z); pd.setXYZ(1, pos.x, 0, pos.z); pd.needsUpdate = true; drop.computeLineDistances();
    }
    function frame(st, instant) {
      const p = toV(st.p), dr = Math.hypot(st.p[0], st.p[1]);
      const want = new T.Vector3(p.x * 0.7, p.y * 0.75, p.z * 0.7);
      if (instant) camS.target.copy(want); else camS.target.lerp(want, 0.08);
      const dist = (70 + 0.85 * p.y + 0.5 * dr) * camS.mul * (cam.aspect < 0.8 ? 1.5 : 1);
      cam.position.set(camS.target.x + dist * Math.cos(camS.el) * Math.sin(camS.az), camS.target.y + dist * Math.sin(camS.el), camS.target.z + dist * Math.cos(camS.el) * Math.cos(camS.az));
      if (cam.position.y < 3) cam.position.y = 3;
      cam.lookAt(camS.target);
      if (rocket) rocket.scale.setScalar(Math.max(1, Math.min(90, cam.position.distanceTo(p) / 22)));
      landRing.scale.setScalar(Math.max(1, dist / 600));
    }
    const apex = () => fl.tr.reduce((a, b) => (b.h > a.h ? b : a), fl.tr[0]);
    function load(_Dz, _fl, wind) { Dz = _Dz; fl = _fl; build(); parts.visible = false; scorch.visible = false; landRing.visible = false; setWind(wind || fl.wind); }
    function showFinal() {
      const last = fl.tr[fl.tr.length - 1]; setTrail(1e9); place(last, 1e9);
      if (fl.outcome === 'APOGEE' && fl.land) { landRing.position.set(fl.land[0], 0.3, -fl.land[1]); landRing.visible = true; }
      const dead = fl.outcome === 'CRASH' || fl.outcome === 'BURST';
      rocket.visible = !dead; if (dead) { scorch.position.set(last.p[0], 0.2, -last.p[1]); scorch.visible = true; }
      frame(apex(), true); dirty = true;
    }
    function start(onFrame, onEnd) {
      rocket.visible = true; parts.visible = false; scorch.visible = false; landRing.visible = false;
      const tEnd = fl.tr[fl.tr.length - 1].t, tA = fl.ev.apo || fl.ev.impact || tEnd;
      const Da = Math.max(4, Math.min(11, tA * 0.45)), Dd = fl.outcome === 'APOGEE' ? 3.5 : Math.max(0.01, Math.min(4, (tEnd - tA) * 0.5));
      play = { t0: performance.now(), pre: opts.countdown === false ? 0 : 1.6, Da, Dd, tA, tEnd, onFrame, onEnd, boomed: false };
    }
    function tick(now) {
      requestAnimationFrame(tick);
      if (play) {
        const e = (now - play.t0) / 1000 - play.pre;
        let ts = e < 0 ? e : e < play.Da ? e / play.Da * play.tA : play.tA + Math.min(1, (e - play.Da) / play.Dd) * (play.tEnd - play.tA);
        if (fl.outcome === 'BURST') ts = Math.min(ts, 0.3);
        const st = sample(fl.tr, Math.max(0, ts));
        setTrail(Math.max(0, ts)); place(st, ts); frame(st, false);
        play.onFrame && play.onFrame(ts, st, e);
        const endE = fl.outcome === 'BURST' ? 0.6 : play.Da + play.Dd;
        if (!play.boomed && ((fl.outcome === 'CRASH' && e >= endE) || (fl.outcome === 'BURST' && e >= 0.3))) { play.boomed = true; rocket.visible = false; boom(toV(st.p), fl.outcome === 'BURST'); }
        if (e >= endE + (play.boomed ? 1.4 : 0)) { const p = play; play = null; if (fl.outcome === 'APOGEE' && fl.land) { landRing.position.set(fl.land[0], 0.3, -fl.land[1]); landRing.visible = true; } p.onEnd && p.onEnd(); }
        dirty = true;
      }
      if (parts.visible) { const pa = pg.attributes.position; for (let i = 0; i < NP; i++) { const v = pvel[i]; if (!v) continue; v[1] -= 9.8 * 0.016; pa.setXYZ(i, pa.getX(i) + v[0] * 0.016, Math.max(0, pa.getY(i) + v[1] * 0.016), pa.getZ(i) + v[2] * 0.016); } pa.needsUpdate = true; pm.opacity = Math.max(0, pm.opacity - 0.006); if (pm.opacity <= 0) parts.visible = false; dirty = true; }
      if (dirty && visible) { if (!play && fl && !parts.visible) frame(apex(), true); renderer.render(scene, cam); dirty = false; }
    }
    requestAnimationFrame(tick);
    return { load, start, showFinal, resize, get playing() { return !!play; } };
  }

  window.Render = { C, fmt, cutSVG, insets, curveSVG, drawSide, sample, aoaAt, Sky, sizeCanvas };
})();
