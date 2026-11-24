/* TAS-1 Launch League — 기체 + 3D 비행 공용 모듈 (브라우저: window.Flight, 서버: require)
 * 엔진 입력은 TAS-1 시계열 대리모델 결과표의 한 행(row): thrust/mdot 곡선, 사용 추진제, 형상.
 * 기체(CG/CP)와 비행은 교육용 단순 모델이다.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Flight = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const AIR = {
    body: { nm: 'Body Tube', ko: '지관 길이', vals: [1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2, 4.8], u: 'm', d: 1 },
    fin: { nm: 'Fin Span', ko: '핀 크기', vals: [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16, 0.19, 0.22], u: 'm', d: 2 },
  };
  const NOSES = [
    { id: 'conical', nm: 'Conical', ko: '원뿔', cp: 0.666, dcd: 0.06 },
    { id: 'ogive', nm: 'Ogive', ko: '오자이브', cp: 0.466, dcd: 0.0 },
    { id: 'ellip', nm: 'Elliptical', ko: '타원', cp: 0.333, dcd: 0.02 },
    { id: 'vonk', nm: 'Von Kármán', ko: '폰 카르만', cp: 0.5, dcd: -0.02 },
  ];
  const D = 0.13, R = D / 2, AREF = Math.PI * R * R, LN = 4 * D;
  const CR = 0.26, CT = 0.10, XSW = 0.16, NFIN = 4, BAY = 0.5, L_INJ = 0.05;
  const RHO_AL = 2700, RHO_FUEL = 950;

  function noseR(id, x) {
    const Ln = LN, r = R;
    if (id === 'conical') return r * x / Ln;
    if (id === 'ogive') { const rho = (r * r + Ln * Ln) / (2 * r); return Math.sqrt(Math.max(0, rho * rho - (Ln - x) * (Ln - x))) + r - rho; }
    if (id === 'ellip') return r * Math.sqrt(Math.max(0, 1 - Math.pow((Ln - x) / Ln, 2)));
    const th = Math.acos(Math.max(-1, Math.min(1, 1 - 2 * x / Ln)));
    return r / Math.sqrt(Math.PI) * Math.sqrt(Math.max(0, th - Math.sin(2 * th) / 2));
  }

  // 엔진 행 → 시간 함수
  function engineFn(row) {
    const dt = row.dt, th = row.thrust, md = row.mdot;
    const n = th.length, tEnd = row.t_end;
    const at = (arr, t) => {
      if (t < 0 || t >= tEnd) return 0;
      const f = t / dt, i = Math.floor(f);
      if (i >= n - 1) return arr[n - 1] || 0;
      return arr[i] + (arr[i + 1] - arr[i]) * (f - i);
    };
    // 누적 소모량 (사다리꼴)
    const cum = [0];
    for (let i = 1; i < n; i++) cum.push(cum[i - 1] + (md[i - 1] + md[i]) / 2 * dt);
    const usedAt = t => {
      if (t <= 0) return 0;
      const tt = Math.min(t, tEnd), f = tt / dt, i = Math.floor(f);
      if (i >= n - 1) return cum[n - 1];
      return cum[i] + (cum[i + 1] - cum[i]) * (f - i);
    };
    const total = cum[n - 1] || 1e-9;
    return { thrust: t => at(th, t), usedAt, total, tEnd };
  }

  function airframe(row, ox, a) {
    const g0 = row.geom;
    const Lb = AIR.body.vals[a.body], s = AIR.fin.vals[a.fin], nose = NOSES[a.nose];
    const wt = row.wall_t || 0.004, rIn = 0.055;
    const ringA = Math.PI * ((rIn + wt) ** 2 - rIn ** 2);
    const Lt = g0.tank_len + 0.06, Lc = g0.chamber_len;
    const Lnz = 0.03 + Math.max(0, g0.exit_r - g0.throat_r) / Math.tan(15 * Math.PI / 180);
    const Le = Lt + L_INJ + Lc + Lnz + 0.04;
    const Ltot = LN + Lb;
    const over = Math.max(0, Le + BAY - Lb);
    const xEnd = Ltot + over;                    // 엔진 꼬리 (조립 불가면 지관 밖으로 튀어나옴)
    const xNz0 = xEnd - Lnz, xC0 = xNz0 - Lc, xI0 = xC0 - L_INJ, xT0 = xI0 - Lt;
    const finA = (CR + CT) / 2 * s;
    const grainA = Math.PI * g0.outer_r ** 2 - g0.port_n * Math.PI * g0.port_r ** 2;
    const fuelLoad = RHO_FUEL * grainA * g0.fuel_len;
    const parts = [
      { k: 'nose', m: 0.5, x: 0.6 * LN },
      { k: 'bay', m: 2.0, x: LN + BAY / 2 },
      { k: 'body', m: 1.1 * Lb, x: LN + Lb / 2 },
      { k: 'fin', m: NFIN * finA * 0.003 * 1850, x: Ltot - CR * 0.55 },
      { k: 'tank', m: RHO_AL * ringA * Lt + 0.6, x: xT0 + Lt / 2 },
      { k: 'inj', m: 0.5, x: xI0 + L_INJ / 2 },
      { k: 'chamber', m: RHO_AL * ringA * Lc + 0.3, x: xC0 + Lc / 2 },
      { k: 'nozzle', m: 0.4 + 8 * g0.exit_r, x: xNz0 + Lnz / 2 },
    ];
    const dry = parts.reduce((t, p) => t + p.m, 0);
    const dryM = parts.reduce((t, p) => t + p.m * p.x, 0);
    // Barrowman
    const Lf = Math.sqrt(s * s + Math.pow(XSW + CT / 2 - CR / 2, 2));
    const cnF = (1 + R / (s + R)) * (4 * NFIN * Math.pow(s / D, 2)) / (1 + Math.sqrt(1 + Math.pow(2 * Lf / (CR + CT), 2)));
    const xb = Ltot - CR;
    const xF = xb + XSW / 3 * (CR + 2 * CT) / (CR + CT) + (CR + CT - CR * CT / (CR + CT)) / 6;
    const xN = nose.cp * LN;
    const cna = 2 + cnF;
    const cp = (2 * xN + cnF * xF) / cna;
    const cd0 = 0.30 + 0.04 * Ltot + nose.dcd + 0.025 * (NFIN * finA / AREF);
    const E = engineFn(row);
    const oxU = row.ox_used, fuU = row.fuel_used, fOx = oxU / Math.max(oxU + fuU, 1e-9);
    function massAt(t) {
      const used = E.usedAt(t) / E.total;           // 0..1 of burnt propellant
      const mOx = Math.max(0, ox - oxU * used);
      const fracLiq = Math.max(0, mOx / ox);
      const xOx = xT0 + Lt - (Lt * 0.85 * fracLiq) / 2;
      const mFuel = Math.max(0, fuelLoad - fuU * used);
      const xFu = xC0 + Lc / 2;
      const m = dry + mOx + mFuel;
      return { m, cg: (dryM + mOx * xOx + mFuel * xFu) / m };
    }
    const m0 = massAt(0);
    return {
      Lb, Ltot, Lt, Lc, Lnz, Le, over, fit: over <= 0, xEnd, xNz0, xC0, xI0, xT0, s, nose, finA, fuelLoad,
      dry, cp, cnF, xF, xN, cna, cd0, massAt, m0: m0.m, cg0: m0.cg, margin: (m0.cg !== m0.cg) ? 0 : (cp - m0.cg) / D, E,
    };
  }

  const MSG = {
    ERR_FIT: '조립 불가 · 엔진이 지관에 안 들어감',
    ERR_INJECTOR_DP: '인젝터 차압 부족 · 연소 불안정',
    ERR_TANK_OVERPRESSURE: '탱크 압력 한계 초과',
    ERR_CHAMBER_OVERPRESSURE: '연소실 압력 한계 초과',
    ERR_TIMEOUT: '40초 안에 연소가 끝나지 않음',
    WARN_UNSTABLE: '불안정 · CP가 CG보다 앞',
    WARN_MARGINAL_STABILITY: '안정 여유 부족 · 1 cal 미만',
    WARN_OVERSTABLE: '과안정 · 바람에 꺾일 수 있음',
    WARN_LOW_THRUST_TO_WEIGHT: '추력/중량 낮음 · 레일 이탈 느림',
    WARN_NO_LIFTOFF: '추력이 무게보다 작음',
    WARN_LOW_CONFIDENCE: 'AI 확신 낮음 · 학습 데이터가 드문 영역',
  };

  function design(row, ox, a) {
    const A = airframe(row, ox, a);
    const errs = (row.codes || []).slice(), warns = [];
    if (!A.fit) errs.unshift('ERR_FIT');
    const peak = Math.max(...row.thrust);
    const tw = row.thrust_avg / (A.m0 * 9.81);
    if (A.margin < 0) warns.push('WARN_UNSTABLE');
    else if (A.margin < 1) warns.push('WARN_MARGINAL_STABILITY');
    else if (A.margin > 3.2) warns.push('WARN_OVERSTABLE');
    if (peak < A.m0 * 9.81) warns.push('WARN_NO_LIFTOFF');
    else if (tw < 2.5) warns.push('WARN_LOW_THRUST_TO_WEIGHT');
    const conf = Math.max(0, Math.min(1, 1 - (row.isp_std / Math.max(Math.abs(row.isp), 20)) * 4));
    if (conf < 0.7) warns.push('WARN_LOW_CONFIDENCE');
    return { row, ox, a, A, errs, warns, tw, peak, conf };
  }

  // 결정적 3D 비행. wind: {spd, dir(불어오는 방위, deg)}
  function fly(Dz, opt) {
    opt = opt || {};
    const A = Dz.A, E = A.E;
    const wind = opt.wind || { spd: 0, dir: 0 };
    const wr = wind.dir * Math.PI / 180;
    const wv = [-Math.sin(wr) * wind.spd, -Math.cos(wr) * wind.spd];
    const railLen = 8, el = (opt.elev || 87) * Math.PI / 180, az = (opt.az || 0) * Math.PI / 180;
    let u = [Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el)];
    const rail = u.slice();
    let p = [0, 0, 0], v = [0, 0, 0], w = [0, 0, 0];
    const add = (a, b, k) => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const nrm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
    const ev = {}, tr = [];
    const rec = (t, chute) => tr.push({ t: +t.toFixed(3), p: [+p[0].toFixed(2), +p[1].toFixed(2), +p[2].toFixed(2)], u: [+u[0].toFixed(4), +u[1].toFixed(4), +u[2].toFixed(4)], v: +Math.hypot(v[0], v[1], v[2]).toFixed(2), h: +p[2].toFixed(2), chute: chute ? 1 : 0 });
    const res = (outcome, extra) => Object.assign({ tr, ev, outcome, apo: 0, maxV: 0, railV: 0, land: null, maxAoa: 0, impactV: 0, wind }, extra || {});
    if (Dz.errs.includes('ERR_FIT')) return res('NO_ASSEMBLY');
    if (Dz.errs.includes('ERR_TANK_OVERPRESSURE') || Dz.errs.includes('ERR_CHAMBER_OVERPRESSURE')) { rec(0); ev.boom = 0.3; return res('BURST'); }
    const chug = Dz.errs.includes('ERR_INJECTOR_DP');
    const burnEnd = chug ? Math.min(1.5, E.tEnd) : E.tEnd;
    const thrust = t => (t < burnEnd ? E.thrust(t) * (chug ? 0.4 : 1) : 0);
    const Ltot = A.Ltot;
    let t = 0, onRail = true, maxQ = 0, maxV = 0, apo = 0, railV = 0, outcome = null, maxAoa = 0, nextRec = 0;
    const dt = 0.004;
    while (t < 300) {
      const T = thrust(t);
      const ma = A.massAt(Math.min(t, burnEnd)), m = ma.m, cg = ma.cg;
      const h = p[2];
      const rho = 1.225 * Math.exp(-Math.max(h, 0) / 8500);
      const wf = Math.pow(Math.max(h, 2) / 10, 0.14);
      const va = [v[0] - wv[0] * wf, v[1] - wv[1] * wf, v[2]];
      const V = Math.hypot(va[0], va[1], va[2]);
      const q = 0.5 * rho * V * V;
      let F = add([0, 0, -9.81 * m], u, T);
      if (onRail) {
        const along = dot(F, rail);
        const sp = Math.hypot(v[0], v[1], v[2]) + Math.max(0, along / m) * dt;
        v = [rail[0] * sp, rail[1] * sp, rail[2] * sp];
        p = add(p, v, dt);
        if (dot(p, rail) >= railLen) { onRail = false; railV = sp; ev.rail = t; }
        if (t > Math.max(2, burnEnd) && sp < 0.05) { outcome = 'NO_LIFTOFF'; break; }
      } else {
        const vh = V > 0.5 ? [va[0] / V, va[1] / V, va[2] / V] : u;
        const ca = Math.max(-1, Math.min(1, dot(u, vh)));
        const aoa = Math.acos(ca), sa = Math.sin(aoa);
        if (!ev.apo && V > 25) maxAoa = Math.max(maxAoa, aoa);  // 정점 부근 저속 구간의 큰 받음각은 자세 상실이 아님
        F = add(F, vh, -q * AREF * A.cd0 * (1 + 0.3 * sa));
        F = add(F, vh, -q * 1.2 * Ltot * D * sa * sa);
        const perp = add(vh, u, -ca), pl = Math.hypot(perp[0], perp[1], perp[2]);
        if (pl > 1e-6) F = add(F, perp, -q * AREF * A.cna * sa * Math.min(1, Math.abs(ca) + 0.2) / pl);
        v = add(v, F, dt / m); p = add(p, v, dt);
        const I = m * Ltot * Ltot / 12;
        if (pl > 1e-6) w = add(w, nrm(crs(u, vh)), q * AREF * A.cna * sa * (A.cp - cg) / I * dt);
        const damp = 0.5 * rho * V * AREF * (2 * Math.pow(A.xN - cg, 2) + A.cnF * Math.pow(A.xF - cg, 2)) * Ltot / I + 0.05;
        w = w.map(c => c * Math.max(0, 1 - damp * dt));
        u = nrm(add(u, crs(w, u), dt));
        if (p[2] <= 0 && t > 0.5) { outcome = 'CRASH'; ev.impact = t; break; }
        if (v[2] < 0 && !ev.apo && p[2] > 5) {
          ev.apo = t; apo = p[2];
          if (maxAoa < 0.8) { outcome = 'APOGEE'; rec(t); break; }
          ev.tumble = 1;  // 자세를 잃은 기체는 낙하산 전개 실패
        }
      }
      if (q > maxQ) { maxQ = q; ev.maxq = t; }
      maxV = Math.max(maxV, Math.hypot(v[0], v[1], v[2]));
      if (ev.bo === undefined && t >= burnEnd) ev.bo = t;
      if (t >= nextRec) { rec(t); nextRec += 0.1; }
      t += dt;
    }
    if (!outcome) outcome = 'TIMEOUT';
    let land = null, impactV = 0;
    if (outcome === 'APOGEE') {
      let q = p.slice(), tt = t;
      while (q[2] > 0) {
        const wf = Math.pow(Math.max(q[2], 2) / 10, 0.14), vd = q[2] > 300 ? 25 : 7;
        q = [q[0] + wv[0] * wf * 0.5, q[1] + wv[1] * wf * 0.5, Math.max(0, q[2] - vd * 0.5)]; tt += 0.5;
        if (Math.round(tt * 2) % 4 === 0 || q[2] === 0) tr.push({ t: +tt.toFixed(2), p: q.map(x => +x.toFixed(1)), u: [0, 0, 1], v: vd, h: +q[2].toFixed(1), chute: 1 });
      }
      land = [q[0], q[1]];
    } else if (outcome === 'CRASH') {
      impactV = Math.hypot(v[0], v[1], v[2]);
      rec(t); tr[tr.length - 1].h = 0; tr[tr.length - 1].p[2] = 0;
      land = [p[0], p[1]];
      apo = tr.reduce((a, r) => Math.max(a, r.h), 0);
    } else {
      apo = tr.reduce((a, r) => Math.max(a, r.h), 0);
      rec(t);
    }
    return res(outcome, { apo: Math.round(apo), maxV, railV, land, maxAoa, impactV, chug });
  }

  function score(Dz, fl, target) {
    if (fl.outcome !== 'APOGEE') return { target: 0, max: 0 };
    const acc = 1000 * Math.exp(-Math.abs(fl.apo - target) / (target / 6));
    const eco = 200 * (1 - Dz.ox / 8);
    return { target: Math.round(acc + eco), max: Math.round(fl.apo) };
  }

  return { AIR, NOSES, D, R, LN, CR, CT, XSW, BAY, L_INJ, noseR, airframe, design, fly, score, MSG };
});
