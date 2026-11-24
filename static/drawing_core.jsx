/* 엔진 형상 도면 공용 코드 (drawing.html · report.html 에서 <script type="text/babel" src> 로 사용)
   형상 JSON → derive() → 시트 컴포넌트 (AssemblySheet, TankSheet, InjectorSheet, GrainSheet, ChamberSheet) */
const { useState, useEffect, useLayoutEffect, useMemo, useRef } = React;

/* =====================================================================
   기본 형상 (rocket.json)
   ===================================================================== */
const DEFAULT = {"TAS-I":{"oxidizer":"NitrousOxide","oxidizer_MW":0.044013,"oxidizer_M":4.49,"pressurant":"Helium","pressurant_MW":0.0040026,"pressurant_M":0.025,"fuel":"HDPE","fuel_a":1e-05,"fuel_n":0.83,"fuel_rho":950,"inner_radius":0.02095,"outter_radius":0.04846,"grain_type":"PORT","port_num":2,"port_center_radius":0.02695,"fin_num":6,"fin_length":0.024,"fin_width":0.006,"fin_tip_width":0,"grid_size":300,"Pa":101325,"temperature":258,"cstar_eta":0.9,"cstar_tau":1e-06,"cstar_startup_deficit":0,"oxidizer_tank_radius":0.05,"oxidizer_tank_length":0.92461,"oxidizer_tank_material_rho":2700,"oxidizer_tank_tickness":0.00695,"injector_oriffice_radius":0.00047,"injector_oriffice_length":0.01415,"injector_oriffice_n":38,"injector_cd":0.65,"combution_chamber_pressure":3000000,"combution_chamber_length":0.87863,"combution_chamber_radius":0.05,"fuel_length":0.52257,"nozzle_throat_radius":0.00871,"nozzle_exit_radius":0.01982,"dt":0.01,"simt":20,"thrust_coeff":0.9}};

/* 가정값 (형상 JSON 에 없는 치수) — 도면 주기에 표시 */
const ASSUME = {
  feedOD: 12.7, feedWall: 0.89,        // 1/2" 배관 (OD 12.7 mm, 0.035" 두께)
  feedLen: 180,                         // 탱크–인젝터 배관 길이 (개략)
  bulkMin: 8,                           // 탱크 벌크헤드 최소 두께 [mm] (= max(2t, 8))
  convHalf: 45, divHalf: 15,            // 노즐 수렴/확대 반각 [deg]
  nozInletFrac: 0.85,                   // 노즐 입구 반지름 / 챔버 내경 반지름
  injAvailFrac: 0.75,                   // 오리피스 배치 가능 반지름 / 챔버 내경 반지름
  linerT: null,                         // 라이너 = 챔버 내경 − 연료 외경
};

/* =====================================================================
   형상 계산 (모든 길이 mm)
   ===================================================================== */
const MM = v => (+v || 0) * 1000;

function holeLayout(n, Ravail, rh) {
  n = Math.max(1, Math.round(n));
  if (n === 1) return { holes: [[0, 0]], rings: [{ r: 0, n: 1 }] };
  for (let m = 1; m <= 40; m++) {
    const pitch = Ravail / m;
    const caps = [];
    for (let j = 1; j <= m; j++) caps.push(Math.max(1, Math.floor(2 * Math.PI * j * 0.85)));
    const total = caps.reduce((a, b) => a + b, 0);
    if (total < n && m < 40) continue;
    // 바깥 고리부터 용량 비례로 채움 (작으면 바깥 고리 하나만)
    let counts = caps.map(c => Math.floor(n * c / total));
    let rest = n - counts.reduce((a, b) => a + b, 0);
    for (let j = m - 1; rest > 0; j = (j - 1 + m) % m) { counts[j]++; rest--; }
    const holes = [], rings = [];
    counts.forEach((k, j) => {
      if (k <= 0) return;
      const r = pitch * (j + 1);
      const off = (j % 2) * Math.PI / k;
      for (let i = 0; i < k; i++) {
        const a = off + 2 * Math.PI * i / k;
        holes.push([r * Math.cos(a), r * Math.sin(a)]);
      }
      rings.push({ r, n: k });
    });
    return { holes, rings, pitch };
  }
}

function derive(c) {
  const g = {};
  g.Rt = MM(c.oxidizer_tank_radius); g.t = MM(c.oxidizer_tank_tickness); g.Lt = MM(c.oxidizer_tank_length);
  g.Rc = MM(c.combution_chamber_radius); g.Lc = MM(c.combution_chamber_length);
  g.Rf = MM(c.outter_radius); g.ri = MM(c.inner_radius); g.Lf = MM(c.fuel_length);
  g.grain = String(c.grain_type || 'PORT').toUpperCase();
  g.nport = Math.max(1, parseInt(c.port_num) || 1);
  g.rpc = g.nport === 1 ? 0 : MM(c.port_center_radius);
  g.fin = { n: Math.max(1, parseInt(c.fin_num) || 6), L: MM(c.fin_length), w: MM(c.fin_width),
            tip: c.fin_tip_width === '' || c.fin_tip_width == null ? MM(c.fin_width) : MM(c.fin_tip_width) };
  g.nOri = Math.max(1, parseInt(c.injector_oriffice_n) || 1); g.rOri = MM(c.injector_oriffice_radius); g.LOri = MM(c.injector_oriffice_length);
  g.Rth = MM(c.nozzle_throat_radius); g.Re = MM(c.nozzle_exit_radius);
  g.pre = 0.5 * g.Rf; g.post = 1.5 * g.Rf;
  g.tb = Math.max(2 * g.t, ASSUME.bulkMin);
  // 인젝터
  g.plateT = g.LOri;
  g.layout = holeLayout(g.nOri, ASSUME.injAvailFrac * g.Rc, g.rOri);
  g.manifold = Math.max(8, 0.25 * g.Rc);            // 매니폴드 깊이
  g.capT = Math.max(g.t, 6);                         // 인젝터 헤드 덮개 두께
  // 노즐
  g.Rin = ASSUME.nozInletFrac * g.Rc;
  const tc = Math.tan(ASSUME.convHalf * Math.PI / 180), td = Math.tan(ASSUME.divHalf * Math.PI / 180);
  g.Lconv = Math.max(g.Rin - g.Rth, 0) / tc;
  g.Lland = Math.max(0.3 * g.Rth, 2);
  g.Ldiv = Math.max(g.Re - g.Rth, 0) / td;
  g.lip = 4;
  g.Lnoz = g.lip + g.Lconv + g.Lland + g.Ldiv;
  // 챔버 내부 위치 (챔버 관 앞끝 = 0)
  g.xPre = g.plateT; g.xGrain = g.xPre + g.pre; g.xPost = g.xGrain + g.Lf; g.xPostEnd = g.xPost + g.post;
  g.xNoz = Math.max(g.Lc - g.Lnoz, g.xPostEnd);
  g.spare = g.xNoz - g.xPostEnd;
  g.nozOver = g.xNoz + g.Lnoz - g.Lc;                // 관 밖으로 나온 노즐 길이 (> 0 이면)
  g.eps = g.Rth > 0 ? (g.Re / g.Rth) ** 2 : 0;
  // 파생값
  g.Vtank = Math.PI * g.Rt ** 2 * g.Lt * 1e-6;       // L
  g.Aori = g.nOri * Math.PI * g.rOri ** 2;            // mm²
  g.Aport = g.grain === 'PORT' ? g.nport * Math.PI * g.ri ** 2 : null;
  g.web = g.grain === 'PORT' ? (g.nport === 1 ? g.Rf - g.ri : Math.min(g.Rf - g.rpc - g.ri, g.nport > 1 ? 2 * g.rpc * Math.sin(Math.PI / g.nport) - 2 * g.ri : Infinity)) : null;
  g.mFuel = g.grain === 'PORT' ? (+c.fuel_rho || 0) * (Math.PI * g.Rf ** 2 - g.nport * Math.PI * g.ri ** 2) * g.Lf * 1e-9 : null;
  g.warn = [];
  if (g.Rf > g.Rc) g.warn.push('연료 외경이 챔버 내경보다 큼');
  if (g.xPostEnd > g.Lc) g.warn.push(`챔버 내부 배치(인젝터+전연소실+연료+후연소실 ${g.xPostEnd.toFixed(1)} mm)가 챔버 길이 ${g.Lc.toFixed(1)} mm 초과`);
  if (g.nozOver > 0.5) g.warn.push(`노즐(${g.Lnoz.toFixed(1)} mm)이 챔버 관 끝에서 ${g.nozOver.toFixed(1)} mm 돌출`);
  if (g.layout && g.layout.pitch && g.layout.pitch < 2 * g.rOri * 1.5) g.warn.push('오리피스 간격이 좁음 (피치 < 1.5 × 오리피스 지름)');
  if (g.grain === 'PORT' && g.web != null && g.web < 2) g.warn.push(`포트 사이 최소 웹 ${g.web.toFixed(2)} mm`);
  return g;
}

/* =====================================================================
   도면 기본 요소 (시트 좌표 mm, A3 420 × 297)
   ===================================================================== */
const W_VIS = 0.5, W_THIN = 0.25, INK = 'var(--dw-ink)', PAPER = 'var(--dw-paper)';
const SCALES = [[5, 1], [2, 1], [1, 1], [1, 2], [1, 2.5], [1, 5], [1, 10], [1, 15], [1, 20], [1, 25], [1, 50]];
function pickScale(w, h, aw, ah) {
  for (const [a, b] of SCALES) { const s = a / b; if (w * s <= aw && h * s <= ah) return { s, label: `${a}:${String(b).replace('.', ',')}` }; }
  return { s: 1 / 50, label: '1:50' };
}
function pickScaleCut(L, cuts, H, aw, ah) {
  for (const [a, b] of SCALES) { const sv = a / b; if (compressedLen(L, cuts, sv) * sv <= aw && H * sv <= ah) return { s: sv, label: `${a}:${String(b).replace('.', ',')}` }; }
  return { s: 1 / 50, label: '1:50' };
}
const tankCut = (g, x0 = 0) => { const k = Math.max(60, 2.5 * (g.Rt + g.t)), L = g.Lt + 2 * g.tb; return g.Lt > 4 * k ? [{ a: x0 + g.tb + k, b: x0 + L - g.tb - k, R: g.Rt + g.t }] : []; };
const grainCut = (g, x0 = 0) => { const k = Math.max(50, 1.5 * g.Rc); return g.Lf > 4 * k ? [{ a: x0 + g.xGrain + k, b: x0 + g.xPost - k, R: g.Rc + g.t }] : []; };
const f1 = v => (Math.abs(v - Math.round(v)) < 0.05 ? Math.round(v).toString() : v.toFixed(1));
const f2 = v => v.toFixed(2);

/* 뷰: 모델 mm → 시트 mm (y 위쪽 +) */
/* cuts: [{a, b, R}] 모델 x 구간 생략 (파단), 생략 자리마다 시트 gap mm 의 틈 */
function view(ox, oy, s, cuts = [], gap = 4) {
  const cs = [...cuts].sort((p, q) => p.a - q.a);
  const X = x => {
    let rem = 0, n = 0;
    for (const c of cs) { if (x > c.a) { rem += Math.min(x, c.b) - c.a; if (x >= c.b) n++; } }
    return ox + (x - rem) * s + n * gap;
  };
  return { X, Y: y => oy - y * s, s, cuts: cs, gap };
}
function compressedLen(L, cuts, s, gap = 4) { return cuts.reduce((a, c) => a - (c.b - c.a), L) + cuts.length * gap / s; }

/* 파단선: 틈을 흰색으로 덮고 양쪽에 물결선 */
function Breaks({ V }) {
  return <g>{V.cuts.map((c, i) => {
    const xa = V.X(c.a), xb = xa + V.gap, yt = V.Y(c.R + 2), yb = V.Y(-c.R - 2), h = yb - yt;
    const wave = x => `M${x},${yt} C${x + 1.6},${yt + h * 0.3} ${x - 1.6},${yt + h * 0.7} ${x},${yb}`;
    return (<g key={i}>
      <rect x={xa + 0.15} y={yt - 0.5} width={V.gap - 0.3} height={h + 1} fill={PAPER}/>
      <path d={wave(xa)} fill="none" stroke={INK} strokeWidth={W_THIN}/><path d={wave(xb)} fill="none" stroke={INK} strokeWidth={W_THIN}/>
    </g>);
  })}</g>;
}

function Arrow({ x, y, ux, uy }) {
  const L = 2.6, Wd = 0.75, nx = -uy, ny = ux;
  return <polygon points={`${x},${y} ${x + ux * L + nx * Wd},${y + uy * L + ny * Wd} ${x + ux * L - nx * Wd},${y + uy * L - ny * Wd}`} fill={INK}/>;
}

/* 선형 치수: (x1,y1)–(x2,y2) 시트 좌표, off = 수직 거리 (+ = 왼쪽 법선 방향) */
function Dim({ x1, y1, x2, y2, off = 8, text, fs = 2.8 }) {
  const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1e-9;
  const ux = dx / L, uy = dy / L, nx = uy, ny = -ux, sg = off >= 0 ? 1 : -1;
  const ax = x1 + nx * off, ay = y1 + ny * off, bx = x2 + nx * off, by = y2 + ny * off;
  let ang = Math.atan2(dy, dx) * 180 / Math.PI;
  if (ang > 90.01 || ang < -89.99) ang += 180;
  const inside = L > 7;
  const mx = (ax + bx) / 2 + nx * sg * 1.1 * (ang === Math.atan2(dy, dx) * 180 / Math.PI ? 1 : 1), my = (ay + by) / 2;
  const tx = (ax + bx) / 2 - (Math.abs(uy) > 0.7 ? 0 : 0), ty = (ay + by) / 2;
  // 글자는 치수선 위쪽 (화면 기준)
  const tOffX = Math.abs(uy) > 0.7 ? -1.2 : 0, tOffY = Math.abs(uy) > 0.7 ? 0 : -1.0;
  return (
    <g>
      <line x1={x1 + nx * sg * 1} y1={y1 + ny * sg * 1} x2={ax + nx * sg * 1.5} y2={ay + ny * sg * 1.5} stroke={INK} strokeWidth={W_THIN * 0.8}/>
      <line x1={x2 + nx * sg * 1} y1={y2 + ny * sg * 1} x2={bx + nx * sg * 1.5} y2={by + ny * sg * 1.5} stroke={INK} strokeWidth={W_THIN * 0.8}/>
      <line x1={inside ? ax : ax - ux * 5} y1={inside ? ay : ay - uy * 5} x2={inside ? bx : bx + ux * 5} y2={inside ? by : by + uy * 5} stroke={INK} strokeWidth={W_THIN * 0.8}/>
      {inside ? <><Arrow x={ax} y={ay} ux={ux} uy={uy}/><Arrow x={bx} y={by} ux={-ux} uy={-uy}/></>
              : <><Arrow x={ax} y={ay} ux={-ux} uy={-uy}/><Arrow x={bx} y={by} ux={ux} uy={uy}/></>}
      <text x={tx + tOffX} y={ty + tOffY} fontSize={fs} textAnchor="middle" fill={INK}
            transform={`rotate(${ang} ${tx + tOffX} ${ty + tOffY})`} className="dt">{text}</text>
    </g>
  );
}

/* 지시선 + 글자 */
function Leader({ x, y, tx, ty, text, fs = 2.6, anchor }) {
  const left = anchor ? anchor === 'end' : tx < x;
  return (
    <g>
      <line x1={x} y1={y} x2={tx} y2={ty} stroke={INK} strokeWidth={W_THIN * 0.8}/>
      <line x1={tx} y1={ty} x2={tx + (left ? -1 : 1) * (text.length * fs * 0.55 + 1)} y2={ty} stroke={INK} strokeWidth={W_THIN * 0.8}/>
      <circle cx={x} cy={y} r={0.45} fill={INK}/>
      <text x={tx + (left ? -1 : 1) * 0.6} y={ty - 0.8} fontSize={fs} textAnchor={left ? 'end' : 'start'} fill={INK} className="dt">{text}</text>
    </g>
  );
}

function CL({ x1, y1, x2, y2 }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={INK} strokeWidth={W_THIN * 0.7} strokeDasharray="8 1.5 1 1.5"/>;
}

function Balloon({ x, y, bx, by, n }) {
  const r = 3.2, d = Math.hypot(bx - x, by - y) || 1;
  return (
    <g>
      <line x1={x} y1={y} x2={bx - (bx - x) / d * r} y2={by - (by - y) / d * r} stroke={INK} strokeWidth={W_THIN}/>
      <circle cx={x} cy={y} r={0.6} fill={INK}/>
      <circle cx={bx} cy={by} r={r} fill={PAPER} stroke={INK} strokeWidth={W_THIN * 1.4}/>
      <text x={bx} y={by + 1.2} fontSize={3.4} textAnchor="middle" fill={INK} className="dt" fontWeight="600">{n}</text>
    </g>
  );
}

function ViewLabel({ x, y, text, sub }) {
  return (
    <g>
      <text x={x} y={y} fontSize={3.6} textAnchor="middle" fill={INK} className="dt" fontWeight="600">{text}</text>
      {sub && <text x={x} y={y + 4.2} fontSize={2.6} textAnchor="middle" fill={INK} className="dt">{sub}</text>}
    </g>
  );
}

function Defs() {
  return (
    <defs>
      <pattern id="h45" patternUnits="userSpaceOnUse" width="1.6" height="1.6" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="1.6" stroke={INK} strokeWidth="0.16"/></pattern>
      <pattern id="h135" patternUnits="userSpaceOnUse" width="1.6" height="1.6" patternTransform="rotate(135)">
        <line x1="0" y1="0" x2="0" y2="1.6" stroke={INK} strokeWidth="0.16"/></pattern>
      <pattern id="hfuel" patternUnits="userSpaceOnUse" width="2.4" height="2.4" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="2.4" stroke={INK} strokeWidth="0.14"/><line x1="0.8" y1="0" x2="0.8" y2="2.4" stroke={INK} strokeWidth="0.07"/></pattern>
      <pattern id="hgraph" patternUnits="userSpaceOnUse" width="1.2" height="1.2" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="1.2" stroke={INK} strokeWidth="0.14"/><line x1="0" y1="0.6" x2="1.2" y2="0.6" stroke={INK} strokeWidth="0.14"/></pattern>
      <pattern id="hliner" patternUnits="userSpaceOnUse" width="3" height="3" patternTransform="rotate(135)">
        <line x1="0" y1="0" x2="0" y2="3" stroke={INK} strokeWidth="0.12"/></pattern>
    </defs>
  );
}

/* 시트: 테두리 · 구역 표시 · 표제란 */
function Sheet({ title, no, scale, name, material, children, notes = [], parts }) {
  const date = new Date().toLocaleDateString('sv-SE');          // 로컬 날짜 YYYY-MM-DD
  const TB = { x: 410 - 180, y: 287 - 40, w: 180, h: 40 };
  const zonesX = 8, zonesY = 6;
  return (
    <svg viewBox="0 0 420 297" xmlns="http://www.w3.org/2000/svg" style={{ fontFamily: "'IBM Plex Sans','IBM Plex Sans KR',sans-serif" }}>
      <style>{`.dt{font-family:'IBM Plex Sans','IBM Plex Sans KR',sans-serif;fill:var(--dw-ink)}`}</style>
      <Defs/>
      <rect x="0" y="0" width="420" height="297" fill={PAPER}/>
      <rect x="5" y="5" width="410" height="287" fill="none" stroke={INK} strokeWidth={W_THIN}/>
      <rect x="10" y="10" width="400" height="277" fill="none" stroke={INK} strokeWidth={0.7}/>
      {Array.from({ length: zonesX }, (_, i) => (
        <g key={'zx' + i}>
          {i > 0 && <><line x1={10 + i * 50} y1={5} x2={10 + i * 50} y2={10} stroke={INK} strokeWidth={W_THIN}/><line x1={10 + i * 50} y1={287} x2={10 + i * 50} y2={292} stroke={INK} strokeWidth={W_THIN}/></>}
          <text x={35 + i * 50} y={8.6} fontSize={2.4} textAnchor="middle" fill={INK} className="dt">{i + 1}</text>
          <text x={35 + i * 50} y={290.6} fontSize={2.4} textAnchor="middle" fill={INK} className="dt">{i + 1}</text>
        </g>))}
      {Array.from({ length: zonesY }, (_, i) => (
        <g key={'zy' + i}>
          {i > 0 && <><line x1={5} y1={10 + i * 46.17} x2={10} y2={10 + i * 46.17} stroke={INK} strokeWidth={W_THIN}/><line x1={410} y1={10 + i * 46.17} x2={415} y2={10 + i * 46.17} stroke={INK} strokeWidth={W_THIN}/></>}
          <text x={7.5} y={34 + i * 46.17} fontSize={2.4} textAnchor="middle" fill={INK} className="dt">{'ABCDEF'[i]}</text>
          <text x={412.5} y={34 + i * 46.17} fontSize={2.4} textAnchor="middle" fill={INK} className="dt">{'ABCDEF'[i]}</text>
        </g>))}
      {children}
      {/* 주기 */}
      {notes.length > 0 && (
        <g>
          <text x={16} y={287 - 6 - notes.length * 3.6 - 2} fontSize={2.8} fill={INK} className="dt" fontWeight="600">주기</text>
          {notes.map((n, i) => <text key={i} x={16} y={287 - 6 - (notes.length - 1 - i) * 3.6} fontSize={2.4} fill={INK} className="dt">{`${i + 1}. ${n}`}</text>)}
        </g>
      )}
      {/* 부품표 */}
      {parts && (() => {
        const rowH = 5, y0 = TB.y - rowH * (parts.length + 1);
        const cols = [0, 12, 70, 140, 180];
        return (
          <g>
            <rect x={TB.x} y={y0} width={TB.w} height={rowH * (parts.length + 1)} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>
            {parts.map((p, i) => (
              <g key={i}>
                <line x1={TB.x} y1={y0 + rowH * (i + 1)} x2={TB.x + TB.w} y2={y0 + rowH * (i + 1)} stroke={INK} strokeWidth={W_THIN}/>
                {p.map((v, j) => <text key={j} x={TB.x + cols[j] + (j === 0 ? 6 : 2)} y={y0 + rowH * (i + 1) - 1.5} fontSize={2.5} textAnchor={j === 0 ? 'middle' : 'start'} fill={INK} className="dt">{v}</text>)}
              </g>))}
            {['번호', '품명', '재질 · 규격', '수량'].map((h, j) => <text key={h} x={TB.x + cols[j] + (j === 0 ? 6 : 2)} y={y0 + rowH * (parts.length + 1) - 1.5} fontSize={2.5} textAnchor={j === 0 ? 'middle' : 'start'} fill={INK} className="dt" fontWeight="600">{h}</text>)}
            <line x1={TB.x} y1={y0 + rowH * parts.length} x2={TB.x + TB.w} y2={y0 + rowH * parts.length} stroke={INK} strokeWidth={W_VIS}/>
            {cols.slice(1, 4).map(cx => <line key={cx} x1={TB.x + cx} y1={y0} x2={TB.x + cx} y2={TB.y} stroke={INK} strokeWidth={W_THIN}/>)}
          </g>
        );
      })()}
      {/* 표제란 */}
      <g>
        <rect x={TB.x} y={TB.y} width={TB.w} height={TB.h} fill={PAPER} stroke={INK} strokeWidth={0.7}/>
        <line x1={TB.x} y1={TB.y + 14} x2={TB.x + TB.w} y2={TB.y + 14} stroke={INK} strokeWidth={W_THIN}/>
        <line x1={TB.x} y1={TB.y + 27} x2={TB.x + TB.w} y2={TB.y + 27} stroke={INK} strokeWidth={W_THIN}/>
        <line x1={TB.x + 60} y1={TB.y + 14} x2={TB.x + 60} y2={TB.y + 40} stroke={INK} strokeWidth={W_THIN}/>
        <line x1={TB.x + 105} y1={TB.y + 14} x2={TB.x + 105} y2={TB.y + 40} stroke={INK} strokeWidth={W_THIN}/>
        <line x1={TB.x + 140} y1={TB.y + 14} x2={TB.x + 140} y2={TB.y + 40} stroke={INK} strokeWidth={W_THIN}/>
        <text x={TB.x + 4} y={TB.y + 5} fontSize={2.2} fill={INK} className="dt">도면명</text>
        <text x={TB.x + 4} y={TB.y + 11.5} fontSize={5} fill={INK} className="dt" fontWeight="600">{title}</text>
        <text x={TB.x + TB.w - 4} y={TB.y + 5} fontSize={2.2} fill={INK} className="dt" textAnchor="end">TUSI · TAS-1 HYBRID ROCKET ENGINE</text>
        <text x={TB.x + TB.w - 4} y={TB.y + 11.5} fontSize={3.2} fill={INK} className="dt" textAnchor="end">{name}</text>
        {[[0, '도번', no], [60, '재질', material || '—'], [105, '축척', scale], [140, '단위', 'mm']].map(([x, k, v]) => (
          <g key={k}><text x={TB.x + x + 3} y={TB.y + 18} fontSize={2.2} fill={INK} className="dt">{k}</text>
            <text x={TB.x + x + 3} y={TB.y + 24.5} fontSize={3.4} fill={INK} className="dt">{v}</text></g>))}
        {[[0, '작성일', date], [60, '투상법', '3각법'], [105, '일반 공차', '±0.1 (가공 치수)'], [140, '시트', no.split('-').pop()]].map(([x, k, v]) => (
          <g key={k}><text x={TB.x + x + 3} y={TB.y + 31} fontSize={2.2} fill={INK} className="dt">{k}</text>
            <text x={TB.x + x + 3} y={TB.y + 37.5} fontSize={3.2} fill={INK} className="dt">{v}</text></g>))}
        {/* 3각법 기호 */}
        <g transform={`translate(${TB.x + 80},${TB.y + 34})`}>
          <polygon points="0,-2.5 7,-4 7,4 0,2.5" fill="none" stroke={INK} strokeWidth={0.3}/>
          <circle cx={13} cy={0} r={4} fill="none" stroke={INK} strokeWidth={0.3}/><circle cx={13} cy={0} r={2.5} fill="none" stroke={INK} strokeWidth={0.3}/>
        </g>
      </g>
    </svg>
  );
}

/* =====================================================================
   부품 그리기 (모델 좌표 mm, x = 축 방향, y = 반지름 방향)
   ===================================================================== */

/* 산화제 탱크 단면 (x0 = 앞 벌크헤드 바깥면) */
function TankSection({ g, V, x0 = 0, ports = true }) {
  const { Rt, t, Lt, tb } = g, Ro = Rt + t;
  const L = Lt + 2 * tb;
  const X = x => V.X(x0 + x), Y = V.Y;
  const portR = 7.0;                   // 1/2" 포트 구멍 반지름
  const rect = (xa, ya, xb, yb, fill) => <rect x={Math.min(X(xa), X(xb))} y={Math.min(Y(ya), Y(yb))} width={Math.abs(X(xb) - X(xa))} height={Math.abs(Y(yb) - Y(ya))} fill={fill} stroke={INK} strokeWidth={W_VIS}/>;
  return (
    <g>
      {/* 관 벽 (위·아래) */}
      {rect(0, Rt, L, Ro, 'url(#h45)')}
      {rect(0, -Rt, L, -Ro, 'url(#h45)')}
      {/* 벌크헤드 (관 안쪽 끼움, 포트 구멍) */}
      {rect(0, ports ? portR : 0, tb, Rt, 'url(#h135)')}
      {rect(0, ports ? -portR : 0, tb, -Rt, 'url(#h135)')}
      {rect(L - tb, portR, L, Rt, 'url(#h135)')}
      {rect(L - tb, -portR, L, -Rt, 'url(#h135)')}
      {!ports && null}
      {/* 내부 */}
      <line x1={X(tb)} y1={Y(Rt)} x2={X(L - tb)} y2={Y(Rt)} stroke={INK} strokeWidth={W_VIS}/>
      <line x1={X(tb)} y1={Y(-Rt)} x2={X(L - tb)} y2={Y(-Rt)} stroke={INK} strokeWidth={W_VIS}/>
      <CL x1={X(-6)} y1={Y(0)} x2={X(L + 6)} y2={Y(0)}/>
    </g>
  );
}

/* 샤워헤드 인젝터 단면 (x0 = 헤드 앞면). 헤드 덮개 + 매니폴드 + 오리피스 판 */
function InjectorSection({ g, V, x0 = 0, showHoles = true }) {
  const { Rc, t, plateT, manifold, capT, layout, rOri } = g, Ro = Rc + t;
  const X = x => V.X(x0 + x), Y = V.Y;
  const inR = ASSUME.feedOD / 2;
  const L = capT + manifold + plateT;
  const P = pts => pts.map(([x, y]) => `${X(x)},${Y(y)}`).join(' ');
  // 덮개: 바깥 반지름 Ro (챔버 관과 같은 외경), 중앙 입구 구멍
  const capTop = P([[0, inR], [capT, inR], [capT, Rc], [capT + manifold, Rc], [capT + manifold, Ro], [0, Ro]]);
  const capBot = P([[0, -inR], [capT, -inR], [capT, -Rc], [capT + manifold, -Rc], [capT + manifold, -Ro], [0, -Ro]]);
  // 오리피스 판: 구멍 위치에서 끊어 그림 (단면 = y 좌표가 |y| 인 구멍들)
  const ys = showHoles ? [...new Set(layout.holes.filter(([hx, hy]) => Math.abs(hy) < 1e-6 || true).map(([hx, hy]) => Math.hypot(hx, hy) < 1e-6 ? 0 : null).filter(v => v !== null))] : [];
  // 단면에 보이는 구멍: 각 고리 반지름 ±r (위·아래)
  const cuts = showHoles ? layout.rings.flatMap(r => r.r === 0 ? [0] : [r.r, -r.r]).sort((a, b) => a - b) : [];
  const segs = [];
  let y = -Rc;
  for (const c of cuts) { if (c - rOri > y) segs.push([y, c - rOri]); y = c + rOri; }
  if (Rc > y) segs.push([y, Rc]);
  const xa = capT + manifold, xb = xa + plateT;
  return (
    <g>
      <polygon points={capTop} fill="url(#h45)" stroke={INK} strokeWidth={W_VIS}/>
      <polygon points={capBot} fill="url(#h45)" stroke={INK} strokeWidth={W_VIS}/>
      {segs.map(([a, b], i) => <rect key={i} x={X(xa)} y={Y(b)} width={X(xb) - X(xa)} height={Y(a) - Y(b)} fill="url(#h135)" stroke={INK} strokeWidth={W_VIS}/>)}
      <CL x1={X(-8)} y1={Y(0)} x2={X(L + 8)} y2={Y(0)}/>
    </g>
  );
}

/* 연료 그레인 단면 형상 (끝면) — 포트/핀 경로 */
function grainHoles(g) {
  if (g.grain === 'PORT') {
    if (g.nport === 1) return { circles: [[0, 0, g.ri]], polys: [] };
    return { circles: Array.from({ length: g.nport }, (_, k) => { const a = 2 * Math.PI * k / g.nport + Math.PI / 2; return [g.rpc * Math.cos(a), g.rpc * Math.sin(a), g.ri]; }), polys: [] };
  }
  // FIN / FINOCYL: 중앙 보어 + 반지름 방향 슬롯
  const polys = [];
  const { n, L, w, tip } = g.fin;
  for (let k = 0; k < n; k++) {
    const a = 2 * Math.PI * k / n + Math.PI / 2, ca = Math.cos(a), sa = Math.sin(a);
    const pts = [[g.ri * 0.9, -w / 2], [g.ri + L, -tip / 2], [g.ri + L, tip / 2], [g.ri * 0.9, w / 2]];
    polys.push(pts.map(([x, y]) => [x * ca - y * sa, x * sa + y * ca]));
  }
  return { circles: [[0, 0, g.ri]], polys };
}

/* 끝면 (원) 뷰 공통 */
function GrainEnd({ g, V, cx, cy, hatch = true }) {
  const s = V.s, { circles, polys } = grainHoles(g);
  return (
    <g>
      <circle cx={cx} cy={cy} r={g.Rf * s} fill={hatch ? 'url(#hfuel)' : PAPER} stroke={INK} strokeWidth={W_VIS}/>
      {circles.map(([x, y, r], i) => <circle key={i} cx={cx + x * s} cy={cy - y * s} r={r * s} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>)}
      {polys.map((p, i) => <polygon key={'p' + i} points={p.map(([x, y]) => `${cx + x * s},${cy - y * s}`).join(' ')} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>)}
      {polys.length > 0 && <circle cx={cx} cy={cy} r={g.ri * s} fill={PAPER} stroke="none"/>}
      <CL x1={cx - g.Rf * s - 5} y1={cy} x2={cx + g.Rf * s + 5} y2={cy}/>
      <CL x1={cx} y1={cy - g.Rf * s - 5} x2={cx} y2={cy + g.Rf * s + 5}/>
    </g>
  );
}

/* 노즐 인서트 단면 (x0 = 인서트 앞면) */
function NozzleSection({ g, V, x0 = 0 }) {
  const { Rc, Rin, Rth, Re, lip, Lconv, Lland, Ldiv } = g;
  const X = x => V.X(x0 + x), Y = V.Y;
  const xs = [0, lip, lip + Lconv, lip + Lconv + Lland, lip + Lconv + Lland + Ldiv];
  const prof = [[xs[0], Rin], [xs[1], Rin], [xs[2], Rth], [xs[3], Rth], [xs[4], Re]];
  const top = [...prof, [xs[4], Rc], [xs[0], Rc]].map(([x, y]) => `${X(x)},${Y(y)}`).join(' ');
  const bot = [...prof, [xs[4], Rc], [xs[0], Rc]].map(([x, y]) => `${X(x)},${Y(-y)}`).join(' ');
  return (
    <g>
      <polygon points={top} fill="url(#hgraph)" stroke={INK} strokeWidth={W_VIS}/>
      <polygon points={bot} fill="url(#hgraph)" stroke={INK} strokeWidth={W_VIS}/>
    </g>
  );
}

/* 연소 챔버 조립 단면 (x0 = 챔버 관 앞끝) — 관 + 인젝터 판 자리 + 라이너 + 연료 + 노즐 */
function ChamberSection({ g, V, x0 = 0, withInjector = true, detail = true }) {
  const { Rc, t, Lc, Rf, xGrain, Lf, xNoz } = g, Ro = Rc + t;
  const X = x => V.X(x0 + x), Y = V.Y;
  const rect = (xa, ya, xb, yb, fill, key) => <rect key={key} x={Math.min(X(xa), X(xb))} y={Math.min(Y(ya), Y(yb))} width={Math.abs(X(xb) - X(xa))} height={Math.abs(Y(yb) - Y(ya))} fill={fill} stroke={INK} strokeWidth={W_VIS}/>;
  const { circles, polys } = grainHoles(g);
  // 측면 단면에서 보이는 포트 경계 (y=0 단면과 만나는 포트의 y 범위)
  const cutBands = [];
  circles.forEach(([cx, cy, r]) => { if (Math.abs(cx) <= r) { const h = Math.sqrt(r * r - cx * cx); cutBands.push([cy - h, cy + h]); } });
  if (g.grain !== 'PORT') polys.forEach(p => {   // 슬롯이 x≈0 단면을 지나면 근사
    const xsP = p.map(q => q[0]); if (Math.min(...xsP) <= 0 && Math.max(...xsP) >= 0) { const yv = p.map(q => q[1]); cutBands.push([Math.min(...yv), Math.max(...yv)]); }
  });
  cutBands.sort((a, b) => a[0] - b[0]);
  const solid = []; let y = -Rf;
  for (const [a, b] of cutBands) { if (a > y) solid.push([y, Math.min(a, Rf)]); y = Math.max(y, b); }
  if (Rf > y) solid.push([y, Rf]);
  return (
    <g>
      {rect(0, Rc, Lc, Ro, 'url(#h45)', 'wt')}
      {rect(0, -Rc, Lc, -Ro, 'url(#h45)', 'wb')}
      {/* 라이너 (연료 외경 ~ 챔버 내경) */}
      {Rc - Rf > 0.2 && <>{rect(xGrain, Rf, xGrain + Lf, Rc, 'url(#hliner)', 'lt')}{rect(xGrain, -Rf, xGrain + Lf, -Rc, 'url(#hliner)', 'lb')}</>}
      {/* 연료 (단면 y=0) */}
      {solid.map(([a, b], i) => rect(xGrain, a, xGrain + Lf, b, 'url(#hfuel)', 'f' + i))}
      {/* 노즐 */}
      <NozzleSection g={g} V={V} x0={x0 + xNoz}/>
      {/* 오리피스 판 자리 (챔버 안 앞끝) */}
      {withInjector && <InjectorPlateOnly g={g} V={V} x0={x0}/>}
      <CL x1={X(-6)} y1={Y(0)} x2={X(Math.max(Lc, xNoz + g.Lnoz) + 6)} y2={Y(0)}/>
    </g>
  );
}
function InjectorPlateOnly({ g, V, x0 }) {
  const { Rc, plateT, layout, rOri } = g;
  const X = x => V.X(x0 + x), Y = V.Y;
  const cuts = layout.rings.flatMap(r => r.r === 0 ? [0] : [r.r, -r.r]).sort((a, b) => a - b);
  const segs = []; let y = -Rc;
  for (const c of cuts) { if (c - rOri > y) segs.push([y, c - rOri]); y = c + rOri; }
  if (Rc > y) segs.push([y, Rc]);
  return <g>{segs.map(([a, b], i) => <rect key={i} x={X(0)} y={Y(b)} width={X(plateT) - X(0)} height={Y(a) - Y(b)} fill="url(#h135)" stroke={INK} strokeWidth={W_VIS}/>)}</g>;
}

/* 1/2" 배관 + 볼 밸브 기호 (x0 → x1, y = 0) */
function FeedLine({ V, xa, xb }) {
  const r = ASSUME.feedOD / 2, ri = r - ASSUME.feedWall;
  const X = V.X, Y = V.Y;
  const mid = (xa + xb) / 2, vh = 9, vw = 10;
  return (
    <g>
      {[[xa, mid - vw / 2], [mid + vw / 2, xb]].map(([a, b], i) => (
        <g key={i}>
          <rect x={X(a)} y={Y(r)} width={X(b) - X(a)} height={Y(ri) - Y(r)} fill="url(#h45)" stroke={INK} strokeWidth={W_THIN * 1.4}/>
          <rect x={X(a)} y={Y(-ri)} width={X(b) - X(a)} height={Y(-r) - Y(-ri)} fill="url(#h45)" stroke={INK} strokeWidth={W_THIN * 1.4}/>
        </g>))}
      {/* 볼 밸브 (나비넥타이 + 손잡이) */}
      <polygon points={`${X(mid - vw / 2)},${Y(vh)} ${X(mid + vw / 2)},${Y(-vh)} ${X(mid + vw / 2)},${Y(vh)} ${X(mid - vw / 2)},${Y(-vh)}`} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>
      <circle cx={X(mid)} cy={Y(0)} r={Math.max(2.2 * V.s, 0.8)} fill={INK}/>
      <line x1={X(mid)} y1={Y(0)} x2={X(mid)} y2={Y(vh + 8)} stroke={INK} strokeWidth={W_VIS}/>
      <line x1={X(mid - 6)} y1={Y(vh + 8)} x2={X(mid + 6)} y2={Y(vh + 8)} stroke={INK} strokeWidth={W_VIS * 1.6}/>
      <CL x1={X(xa)} y1={Y(0)} x2={X(xb)} y2={Y(0)}/>
    </g>
  );
}

/* =====================================================================
   시트들
   ===================================================================== */
function commonNotes(g) {
  return [
    `전연소실 길이 = 연료 외경 반지름 × 0.5 = ${f1(g.pre)} · 후연소실 길이 = 연료 외경 반지름 × 1.5 = ${f1(g.post)}`,
    `가정값: 탱크 벌크헤드 두께 max(2t, ${ASSUME.bulkMin}) = ${f1(g.tb)} · 노즐 수렴 ${ASSUME.convHalf}° / 확대 ${ASSUME.divHalf}° 원추, 입구 Ø = 챔버 내경 × ${ASSUME.nozInletFrac}`,
  ];
}

function AssemblySheet({ g, cfg, name }) {
  const L1 = g.Lt + 2 * g.tb, Lfeed = ASSUME.feedLen, Linj = g.capT + g.manifold;
  const Lch = Math.max(g.Lc, g.xNoz + g.Lnoz);
  const total = L1 + Lfeed + Linj + Lch;
  const H = 2 * Math.max(g.Rt + g.t, g.Rc + g.t) + 40;
  const xTank = 0, xFeed0 = L1, xInj = L1 + Lfeed, xCh = xInj + Linj;
  const cuts = [...tankCut(g, 0), ...grainCut(g, xCh)];
  const sc = pickScaleCut(total, cuts, H, 370, 110);
  const s = sc.s;
  const ox = 210 - compressedLen(total, cuts, s) * s / 2, oy = 100;
  const V = view(ox, oy, s, cuts);
  const kT = cuts.length ? Math.max(60, 2.5 * (g.Rt + g.t)) : L1 * 0.3, kC = Math.max(50, 1.5 * g.Rc);
  const Ro = g.Rt + g.t, Rco = g.Rc + g.t;
  const parts = [
    ['1', '산화제 탱크', `${cfg.oxidizer || ''} · Ø${f1(2 * Ro)}×t${f1(g.t)} 관`, '1'],
    ['2', '1/2" 공급 배관 · 볼 밸브', `OD 12.7 × t0.89`, '1'],
    ['3', '샤워헤드 인젝터', `오리피스 ${g.nOri}×Ø${f2(2 * g.rOri)}`, '1'],
    ['4', '연료 그레인', `${cfg.fuel || ''} · Ø${f1(2 * g.Rf)}×${f1(g.Lf)}`, '1'],
    ['5', '연소 챔버', `Ø${f1(2 * Rco)}×t${f1(g.t)} 관 · L${f1(g.Lc)}`, '1'],
    ['6', '노즐', `흑연 · Ø${f2(2 * g.Rth)} / Ø${f2(2 * g.Re)}`, '1'],
  ].reverse();
  return (
    <Sheet title="엔진 조립도" no="TAS-1-000" scale={sc.label} name={name} material="—" parts={parts}
           notes={[...commonNotes(g), `산화제 탱크–인젝터 연결: 1/2" 배관 (개략 길이 ${Lfeed}), 볼 밸브 위치는 개략 · 물결선 = 길이 방향 생략 (치수는 실제 값)`]}>
      <ViewLabel x={210} y={26} text="조립 단면도" sub={`축척 ${sc.label}`}/>
      <TankSection g={g} V={V} x0={xTank}/>
      <FeedLine V={V} xa={xFeed0} xb={xInj}/>
      <InjectorSection g={g} V={V} x0={xInj}/>
      <ChamberSection g={g} V={V} x0={xCh} withInjector={true}/>
      <Breaks V={V}/>
      {/* 전체 길이 */}
      <Dim x1={V.X(0)} y1={V.Y(-Math.max(Ro, Rco))} x2={V.X(xCh + Lch)} y2={V.Y(-Math.max(Ro, Rco))} off={-14} text={`${f1(total)} (전체 길이, 배관 개략)`}/>
      <Dim x1={V.X(0)} y1={V.Y(-Ro)} x2={V.X(L1)} y2={V.Y(-Ro)} off={-6} text={f1(L1)}/>
      <Dim x1={V.X(xCh)} y1={V.Y(-Rco)} x2={V.X(xCh + g.Lc)} y2={V.Y(-Rco)} off={-6} text={f1(g.Lc)}/>
      <Dim x1={V.X(0)} y1={V.Y(-Ro)} x2={V.X(0)} y2={V.Y(Ro)} off={8} text={`Ø${f1(2 * Ro)}`}/>
      {/* 풍선 */}
      <Balloon x={V.X(g.tb + kT * 0.5)} y={V.Y(Ro)} bx={V.X(g.tb + kT * 0.5) - 8} by={V.Y(Ro) - 16} n="1"/>
      <Balloon x={V.X(xFeed0 + Lfeed * 0.25)} y={V.Y(ASSUME.feedOD / 2)} bx={V.X(xFeed0 + Lfeed * 0.25)} by={V.Y(Ro) - 16} n="2"/>
      <Balloon x={V.X(xInj + g.capT * 0.5)} y={V.Y(g.Rc + g.t * 0.5)} bx={V.X(xInj) - 4} by={V.Y(Rco) - 22} n="3"/>
      <Balloon x={V.X(xCh + g.xGrain + kC * 0.5)} y={V.Y(g.Rf * 0.5)} bx={V.X(xCh + g.xGrain + kC * 0.5)} by={V.Y(Rco) - 16} n="4"/>
      <Balloon x={V.X(xCh + g.xPost + g.post * 0.5)} y={V.Y(g.Rc + g.t * 0.5)} bx={V.X(xCh + g.xPost + g.post * 0.5) + 6} by={V.Y(Rco) - 16} n="5"/>
      <Balloon x={V.X(xCh + g.xNoz + g.Lnoz * 0.6)} y={V.Y((g.Rc + g.Re) / 2)} bx={V.X(xCh + Lch) + 8} by={V.Y(Rco) - 16} n="6"/>
      {/* 하단: 측면 외형 개략 + 주요 제원 */}
      <SpecTable x={16} y={160} rows={[
        ['산화제 / 질량', `${cfg.oxidizer} / ${(+cfg.oxidizer_M).toFixed(2)} kg`],
        ['가압제 / 질량', `${cfg.pressurant} / ${(+cfg.pressurant_M * 1000).toFixed(0)} g`],
        ['탱크 내부 부피', `${g.Vtank.toFixed(2)} L`],
        ['연료 / 질량 (초기)', `${cfg.fuel} / ${g.mFuel != null ? g.mFuel.toFixed(2) + ' kg' : '—'}`],
        ['오리피스 총 면적', `${g.Aori.toFixed(1)} mm² (Cd ${cfg.injector_cd})`],
        ['노즐 목 / 출구 Ø', `${f2(2 * g.Rth)} / ${f2(2 * g.Re)} mm`],
        ['면적비 ε', g.eps.toFixed(2)],
      ]}/>
    </Sheet>
  );
}

function SpecTable({ x, y, rows, title = '주요 제원' }) {
  const rh = 5, w1 = 38, w2 = 52;
  return (
    <g>
      <text x={x} y={y - 2} fontSize={2.8} fill={INK} className="dt" fontWeight="600">{title}</text>
      <rect x={x} y={y} width={w1 + w2} height={rh * rows.length} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>
      <line x1={x + w1} y1={y} x2={x + w1} y2={y + rh * rows.length} stroke={INK} strokeWidth={W_THIN}/>
      {rows.map(([k, v], i) => (
        <g key={k}>
          {i > 0 && <line x1={x} y1={y + rh * i} x2={x + w1 + w2} y2={y + rh * i} stroke={INK} strokeWidth={W_THIN}/>}
          <text x={x + 2} y={y + rh * i + 3.5} fontSize={2.4} fill={INK} className="dt">{k}</text>
          <text x={x + w1 + 2} y={y + rh * i + 3.5} fontSize={2.4} fill={INK} className="dt">{v}</text>
        </g>))}
    </g>
  );
}

function TankSheet({ g, cfg, name }) {
  const Ro = g.Rt + g.t, L = g.Lt + 2 * g.tb;
  const cuts = tankCut(g, 0), kT = Math.max(60, 2.5 * Ro);
  const sc = pickScaleCut(L + 40, cuts, 2 * Ro + 60, 270, 140);
  const s = sc.s;
  const Lv = compressedLen(L, cuts, s);
  const V = view(30, 100, s, cuts);
  const ex = 30 + Lv * s + 40 + Ro * s, ey = 100;    // 끝면 뷰 중심
  const fitEnd = ex + Ro * s < 400 ? 1 : 0;
  return (
    <Sheet title="산화제 탱크" no="TAS-1-001" scale={sc.label} name={name} material={`관 (${cfg.oxidizer_tank_material_rho} kg/m³)`}
           notes={[`물결선 = 길이 방향 생략 (치수는 실제 값) · 벌크헤드 두께 ${f1(g.tb)} 은 가정값 (형상 JSON 에 없음) · 앞쪽 포트 = 충진/벤트, 뒤쪽 포트 = 1/2" 공급 배관`,
                   `내부 부피 ${g.Vtank.toFixed(2)} L · 산화제 ${cfg.oxidizer} ${(+cfg.oxidizer_M).toFixed(2)} kg · 가압제 ${cfg.pressurant} ${(+cfg.pressurant_M * 1000).toFixed(0)} g · 초기 온도 ${cfg.temperature} K`,
                   `관 두께 ${f2(g.t)} = 탱크·챔버 공통 가공 두께 (oxidizer_tank_tickness)`]}>
      <ViewLabel x={30 + Lv * s / 2} y={30} text="단면 A-A" sub={`축척 ${sc.label}`}/>
      <TankSection g={g} V={V} x0={0}/>
      <Breaks V={V}/>
      <Dim x1={V.X(0)} y1={V.Y(-Ro)} x2={V.X(L)} y2={V.Y(-Ro)} off={-16} text={`${f1(L)} (전체)`}/>
      <Dim x1={V.X(g.tb)} y1={V.Y(-Ro)} x2={V.X(L - g.tb)} y2={V.Y(-Ro)} off={-8} text={`${f1(g.Lt)} (내부)`}/>
      <Dim x1={V.X(0)} y1={V.Y(Ro)} x2={V.X(g.tb)} y2={V.Y(Ro)} off={8} text={f1(g.tb)}/>
      <Dim x1={V.X(L) + 0} y1={V.Y(-Rt_(g))} x2={V.X(L)} y2={V.Y(g.Rt)} off={-10} text={`Ø${f1(2 * g.Rt)}`}/>
      <Dim x1={V.X(L)} y1={V.Y(-Ro)} x2={V.X(L)} y2={V.Y(Ro)} off={-20} text={`Ø${f1(2 * Ro)}`}/>
      <Leader x={V.X(g.tb + kT * 0.6)} y={V.Y(g.Rt + g.t / 2)} tx={V.X(g.tb + kT * 0.6) + 14} ty={V.Y(Ro) - 12} text={`t ${f2(g.t)}`}/>
      <Leader x={V.X(L - g.tb / 2)} y={V.Y(3)} tx={V.X(L) + 10} ty={V.Y(Ro) - 6} text={`포트 Ø14 (1/2" 배관)`}/>
      {fitEnd ? (
        <g>
          <ViewLabel x={ex} y={30} text="끝면 (뒤쪽)" sub={`축척 ${sc.label}`}/>
          <circle cx={ex} cy={ey} r={Ro * s} fill="url(#h135)" stroke={INK} strokeWidth={W_VIS}/>
          <circle cx={ex} cy={ey} r={g.Rt * s} fill="none" stroke={INK} strokeWidth={W_THIN} strokeDasharray="2 1"/>
          <circle cx={ex} cy={ey} r={7 * s} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>
          <CL x1={ex - Ro * s - 5} y1={ey} x2={ex + Ro * s + 5} y2={ey}/><CL x1={ex} y1={ey - Ro * s - 5} x2={ex} y2={ey + Ro * s + 5}/>
        </g>) : null}
      <SpecTable x={30} y={175} rows={[
        ['내경 / 외경', `Ø${f1(2 * g.Rt)} / Ø${f1(2 * Ro)}`], ['관 두께', f2(g.t)], ['내부 길이', f1(g.Lt)],
        ['내부 부피', `${g.Vtank.toFixed(2)} L`], ['산화제', `${cfg.oxidizer} ${(+cfg.oxidizer_M).toFixed(2)} kg`],
        ['가압제', `${cfg.pressurant} ${(+cfg.pressurant_M * 1000).toFixed(0)} g`]]}/>
    </Sheet>
  );
}
const Rt_ = g => g.Rt;

function InjectorSheet({ g, cfg, name }) {
  const Ro = g.Rc + g.t, Lh = g.capT + g.manifold + g.plateT;
  const sc = pickScale(2 * Ro * 2 + Lh + 80, 2 * Ro + 60, 330, 160);
  const s = sc.s;
  const fx = 40 + Ro * s + 10, fy = 110;           // 정면 (판 면)
  const V = view(fx + Ro * s + 50, fy, s);          // 단면
  const holes = g.layout.holes;
  return (
    <Sheet title="샤워헤드 인젝터" no="TAS-1-002" scale={sc.label} name={name} material="SUS304 (가정)"
           notes={[`오리피스 ${g.nOri}개 · Ø${f2(2 * g.rOri)} · 길이(판 두께) ${f2(g.plateT)} · L/D ${(g.plateT / (2 * g.rOri)).toFixed(1)} · 총 면적 ${g.Aori.toFixed(1)} mm² · Cd ${cfg.injector_cd}`,
                   `오리피스 배치: 동심원 ${g.layout.rings.length}열 (배치 반지름 ≤ 챔버 내경 × ${ASSUME.injAvailFrac}, 인접 열 엇갈림) — 배치는 도면용 자동 생성`,
                   `헤드 덮개 두께 ${f1(g.capT)} · 매니폴드 깊이 ${f1(g.manifold)} 는 가정값 · 입구 1/2" 배관`]}>
      <ViewLabel x={fx} y={32} text="정면 (오리피스 판, 챔버 쪽)" sub={`축척 ${sc.label}`}/>
      <circle cx={fx} cy={fy} r={g.Rc * s} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>
      <circle cx={fx} cy={fy} r={ASSUME.injAvailFrac * g.Rc * s} fill="none" stroke={INK} strokeWidth={W_THIN * 0.6} strokeDasharray="1 1"/>
      {g.layout.rings.filter(r => r.r > 0).map((r, i) => <circle key={i} cx={fx} cy={fy} r={r.r * s} fill="none" stroke={INK} strokeWidth={W_THIN * 0.7} strokeDasharray="8 1.5 1 1.5"/>)}
      {holes.map(([x, y], i) => <circle key={i} cx={fx + x * s} cy={fy - y * s} r={Math.max(g.rOri * s, 0.25)} fill={PAPER} stroke={INK} strokeWidth={W_THIN * 1.2}/>)}
      <CL x1={fx - g.Rc * s - 6} y1={fy} x2={fx + g.Rc * s + 6} y2={fy}/><CL x1={fx} y1={fy - g.Rc * s - 6} x2={fx} y2={fy + g.Rc * s + 6}/>
      <Dim x1={fx - g.Rc * s} y1={fy + g.Rc * s} x2={fx + g.Rc * s} y2={fy + g.Rc * s} off={-10} text={`Ø${f1(2 * g.Rc)} (= 챔버 내경)`}/>
      {g.layout.rings.map((r, i) => r.r > 0 && (
        <Leader key={i} x={fx + r.r * s * Math.cos(-0.6 - i * 0.35)} y={fy - r.r * s * Math.sin(-0.6 - i * 0.35)}
                tx={fx + g.Rc * s + 12} ty={fy + 6 + i * 6} text={`PCD Ø${f1(2 * r.r)} · ${r.n}×Ø${f2(2 * g.rOri)}`} anchor="start"/>))}
      <ViewLabel x={V.X(Lh / 2)} y={32} text="단면 B-B" sub={`축척 ${sc.label}`}/>
      <InjectorSection g={g} V={V} x0={0}/>
      <Dim x1={V.X(g.capT + g.manifold)} y1={V.Y(-Ro)} x2={V.X(Lh)} y2={V.Y(-Ro)} off={-8} text={f2(g.plateT)}/>
      <Dim x1={V.X(0)} y1={V.Y(-Ro)} x2={V.X(Lh)} y2={V.Y(-Ro)} off={-16} text={f1(Lh)}/>
      <Dim x1={V.X(Lh)} y1={V.Y(-Ro)} x2={V.X(Lh)} y2={V.Y(Ro)} off={-12} text={`Ø${f1(2 * Ro)}`}/>
      <Leader x={V.X(g.capT / 2)} y={V.Y(0)} tx={V.X(0) - 10} ty={V.Y(Ro) - 8} text={`입구 1/2" (Ø${f1(ASSUME.feedOD)})`}/>
      <Leader x={V.X(g.capT + g.manifold / 2)} y={V.Y(g.Rc * 0.5)} tx={V.X(Lh) + 14} ty={V.Y(Ro) - 6} text="매니폴드"/>
      <SpecTable x={40} y={190} rows={[
        ['오리피스 수', `${g.nOri}`], ['오리피스 Ø / 길이', `${f2(2 * g.rOri)} / ${f2(g.plateT)}`], ['L/D', (g.plateT / (2 * g.rOri)).toFixed(1)],
        ['총 유로 면적', `${g.Aori.toFixed(1)} mm²`], ['방출 계수 Cd', `${cfg.injector_cd}`], ['열 수', `${g.layout.rings.length}`]]}/>
    </Sheet>
  );
}

function GrainSheet({ g, cfg, name }) {
  const kG = Math.max(40, g.Rf);
  const cuts = g.Lf > 4 * kG ? [{ a: kG, b: g.Lf - kG, R: g.Rf }] : [];
  const sc = pickScaleCut(g.Lf + 2 * g.Rf + 60, cuts, 2 * g.Rf + 70, 330, 140);
  const s = sc.s;
  const Lv = compressedLen(g.Lf, cuts, s);
  const V = view(40, 105, s, cuts);
  const ex = 40 + Lv * s + 45 + g.Rf * s, ey = 105;
  const { circles, polys } = grainHoles(g);
  // 측면: 외형 + 숨은선 (포트 위아래 경계)
  const hid = [];
  circles.forEach(([cx, cy, r]) => { hid.push(cy + r, cy - r); });
  const isPort = g.grain === 'PORT';
  return (
    <Sheet title="연료 그레인" no="TAS-1-003" scale={sc.label} name={name} material={`${cfg.fuel} (ρ ${cfg.fuel_rho} kg/m³)`}
           notes={[`그레인 형식 ${g.grain}${isPort ? ` · 포트 ${g.nport}개 · Ø${f2(2 * g.ri)}${g.nport > 1 ? ` · PCD Ø${f2(2 * g.rpc)}` : ''}` : ` · 핀 ${g.fin.n}개 · 길이 ${f2(g.fin.L)} · 폭 ${f2(g.fin.w)}`}`,
                   isPort ? `초기 포트 면적 ${g.Aport.toFixed(1)} mm² · 최소 웹 ${g.web.toFixed(2)} · 초기 질량 ${g.mFuel.toFixed(3)} kg` : '핀 형상은 개략 (사다리꼴 슬롯)',
                   `라이너: 연료 외경 Ø${f1(2 * g.Rf)} ~ 챔버 내경 Ø${f1(2 * g.Rc)} (두께 ${f2(g.Rc - g.Rf)})`]}>
      <ViewLabel x={40 + Lv * s / 2} y={32} text="측면" sub={`축척 ${sc.label}`}/>
      <rect x={V.X(0)} y={V.Y(g.Rf)} width={V.X(g.Lf) - V.X(0)} height={2 * g.Rf * s} fill={PAPER} stroke={INK} strokeWidth={W_VIS}/>
      {hid.map((y, i) => <line key={i} x1={V.X(0)} y1={V.Y(y)} x2={V.X(g.Lf)} y2={V.Y(y)} stroke={INK} strokeWidth={W_THIN} strokeDasharray="2 1"/>)}
      <CL x1={V.X(-6)} y1={V.Y(0)} x2={V.X(g.Lf + 6)} y2={V.Y(0)}/>
      <Breaks V={V}/>
      <Dim x1={V.X(0)} y1={V.Y(-g.Rf)} x2={V.X(g.Lf)} y2={V.Y(-g.Rf)} off={-10} text={f1(g.Lf)}/>
      <Dim x1={V.X(0)} y1={V.Y(-g.Rf)} x2={V.X(0)} y2={V.Y(g.Rf)} off={10} text={`Ø${f1(2 * g.Rf)}`}/>
      <ViewLabel x={ex} y={32} text="단면 C-C" sub={`축척 ${sc.label}`}/>
      <GrainEnd g={g} V={V} cx={ex} cy={ey}/>
      {isPort && circles.length > 0 && <Leader x={ex + circles[0][0] * s + g.ri * s * 0.7} y={ey - circles[0][1] * s - g.ri * s * 0.7}
              tx={ex + g.Rf * s + 10} ty={ey - g.Rf * s - 4} text={`${g.nport}×Ø${f2(2 * g.ri)}`} anchor="start"/>}
      {isPort && g.nport > 1 && <circle cx={ex} cy={ey} r={g.rpc * s} fill="none" stroke={INK} strokeWidth={W_THIN * 0.7} strokeDasharray="8 1.5 1 1.5"/>}
      {isPort && g.nport > 1 && <Leader x={ex + g.rpc * s * Math.cos(0.4)} y={ey + g.rpc * s * Math.sin(0.4)} tx={ex + g.Rf * s + 10} ty={ey + g.Rf * s * 0.6} text={`PCD Ø${f2(2 * g.rpc)}`} anchor="start"/>}
      {!isPort && <Leader x={ex} y={ey - g.ri * s} tx={ex + g.Rf * s + 10} ty={ey - g.Rf * s - 4} text={`보어 Ø${f2(2 * g.ri)} · 핀 ${g.fin.n}×${f2(g.fin.L)}`} anchor="start"/>}
      <Dim x1={ex - g.Rf * s} y1={ey + g.Rf * s} x2={ex + g.Rf * s} y2={ey + g.Rf * s} off={-10} text={`Ø${f1(2 * g.Rf)}`}/>
      <SpecTable x={40} y={185} rows={[
        ['연료', `${cfg.fuel}`], ['밀도', `${cfg.fuel_rho} kg/m³`], ['외경 × 길이', `Ø${f1(2 * g.Rf)} × ${f1(g.Lf)}`],
        ['포트', isPort ? `${g.nport} × Ø${f2(2 * g.ri)}` : `${g.grain} (핀 ${g.fin.n})`],
        ['초기 포트 면적', isPort ? `${g.Aport.toFixed(1)} mm²` : '—'], ['초기 질량', isPort ? `${g.mFuel.toFixed(3)} kg` : '—'],
        ['회귀식', `a ${cfg.fuel_a} · n ${cfg.fuel_n}`]]}/>
    </Sheet>
  );
}

function ChamberSheet({ g, cfg, name }) {
  const Ro = g.Rc + g.t, Lall = Math.max(g.Lc, g.xNoz + g.Lnoz);
  const cuts = grainCut(g, 0);
  const sc = pickScaleCut(Lall + 40, cuts, 2 * Ro + 70, 310, 115);
  const s = sc.s;
  const Lv = compressedLen(Lall, cuts, s);
  const V = view(30, 100, s, cuts);
  // 노즐 상세 (확대)
  const dsc = pickScale(g.Lnoz + 20, 2 * g.Rc + 20, 110, 70);
  const DV = view(250, 195, dsc.s);
  const st = [['인젝터 판', 0, g.xPre], ['전연소실', g.xPre, g.xGrain], ['연료', g.xGrain, g.xPost], ['후연소실', g.xPost, g.xPostEnd]];
  if (g.spare > 0.5) st.push(['잔여', g.xPostEnd, g.xNoz]);
  st.push(['노즐', g.xNoz, g.xNoz + g.Lnoz]);
  return (
    <Sheet title="연소 챔버 (노즐 포함)" no="TAS-1-004" scale={sc.label} name={name} material="관 · 노즐 흑연 (가정)"
           notes={[...commonNotes(g), `노즐 면적비 ε = (Re/Rt)² = ${g.eps.toFixed(2)} · 수렴 ${f1(g.Lconv)} + 목 ${f1(g.Lland)} + 확대 ${f1(g.Ldiv)} + 입구 ${g.lip}`,
                   g.spare > 0.5 ? `후연소실 뒤 잔여 ${f1(g.spare)} = 챔버 길이 − (인젝터 판 + 전연소실 + 연료 + 후연소실 + 노즐)` : (g.nozOver > 0.5 ? `노즐이 챔버 관 끝에서 ${f1(g.nozOver)} 돌출` : '챔버 내부 배치 = 챔버 길이')]}>
      <ViewLabel x={30 + Lv * s / 2} y={26} text="단면 D-D" sub={`축척 ${sc.label}`}/>
      <ChamberSection g={g} V={V} x0={0} withInjector={true}/>
      <Breaks V={V}/>
      {st.map(([n, a, b], i) => (b - a) * s > 0.8 && (
        <Dim key={n} x1={V.X(a)} y1={V.Y(Ro)} x2={V.X(b)} y2={V.Y(Ro)} off={8 + (i % 2) * 6} text={(b - a) * s > 14 ? `${n} ${f1(b - a)}` : f1(b - a)}/>))}
      <Dim x1={V.X(0)} y1={V.Y(-Ro)} x2={V.X(g.Lc)} y2={V.Y(-Ro)} off={-10} text={`${f1(g.Lc)} (챔버 관)`}/>
      <Dim x1={V.X(0)} y1={V.Y(-Ro)} x2={V.X(0)} y2={V.Y(Ro)} off={10} text={`Ø${f1(2 * Ro)}`}/>
      <Dim x1={V.X(g.xGrain - 1)} y1={V.Y(-g.Rc)} x2={V.X(g.xGrain - 1)} y2={V.Y(g.Rc)} off={-6} text={`Ø${f1(2 * g.Rc)}`}/>
      {/* 노즐 상세 */}
      <ViewLabel x={260 + g.Lnoz * dsc.s / 2} y={DV.Y(g.Rc) - 14} text="상세 E (노즐)" sub={`축척 ${dsc.label}`}/>
      <NozzleSection g={g} V={DV} x0={0}/>
      <CL x1={DV.X(-5)} y1={DV.Y(0)} x2={DV.X(g.Lnoz + 5)} y2={DV.Y(0)}/>
      <Leader x={DV.X(g.lip + g.Lconv + g.Lland / 2)} y={DV.Y(-g.Rth)} tx={DV.X(g.lip + g.Lconv + g.Lland / 2) - 14} ty={DV.Y(-g.Rc) + 10} text={`목 Ø${f2(2 * g.Rth)}`} anchor="end"/>
      <Dim x1={DV.X(g.Lnoz)} y1={DV.Y(-g.Re)} x2={DV.X(g.Lnoz)} y2={DV.Y(g.Re)} off={-8} text={`Ø${f2(2 * g.Re)}`}/>
      <Dim x1={DV.X(0)} y1={DV.Y(-g.Rc)} x2={DV.X(g.Lnoz)} y2={DV.Y(-g.Rc)} off={-8} text={f1(g.Lnoz)}/>
      <text x={DV.X(g.lip + g.Lconv * 0.5)} y={DV.Y(g.Rc) - 2} fontSize={2.4} textAnchor="middle" fill={INK} className="dt">{ASSUME.convHalf}°</text>
      <text x={DV.X(g.lip + g.Lconv + g.Lland + g.Ldiv * 0.6)} y={DV.Y(g.Rc) - 2} fontSize={2.4} textAnchor="middle" fill={INK} className="dt">{ASSUME.divHalf}°</text>
      <SpecTable x={30} y={185} rows={[
        ['챔버 내경 / 외경', `Ø${f1(2 * g.Rc)} / Ø${f1(2 * Ro)}`], ['챔버 관 길이', f1(g.Lc)],
        ['전연소실 / 후연소실', `${f1(g.pre)} / ${f1(g.post)}`], ['노즐 목 / 출구', `Ø${f2(2 * g.Rth)} / Ø${f2(2 * g.Re)}`],
        ['면적비 ε', g.eps.toFixed(2)], ['설계 챔버압', `${(+cfg.combution_chamber_pressure / 1e6).toFixed(2)} MPa`]]}/>
    </Sheet>
  );
}
