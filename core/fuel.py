"""
fuel_levelset.py
────────────────────────────────────────────────────────────────────────────
하이브리드 로켓 엔진 연료 그레인 레벨셋 시뮬레이션 모듈

기존 단일 원형 그레인 Fuel 클래스를 레벨셋(Level-Set) 기반으로 업그레이드.
  - 포트 개수, 포트 반지름, 외부 반지름을 입력받아 다중 포트 그레인 초기화
  - dt와 후퇴율(rdot)을 입력받아 레벨셋을 전진시킨 후 연소 표면 둘레를 선적분하여 반환
  - 정기 스냅샷 수집 → 웹 시각화용 JSON 직렬화 가능 포맷으로 반환

의존 라이브러리:
    numpy, scipy, scikit-image (skimage)
    pip install numpy scipy scikit-image
"""

import math
import numpy as np
from scipy.ndimage import distance_transform_edt


class Fuel:
    """
    하이브리드 로켓 연료 그레인 클래스 (레벨셋 기반 다중 포트 / 멀티핀 지원)

    config 딕셔너리 키:
        필수 키
        ─────────────────────────────────────────────────────
        NAME            str     연료 종류 이름
        LENGTH          float   그레인 길이 [m]
        INNER_RADIUS    float   포트(가스 통로) 반지름 [m] (FIN 타입에서는 중앙 보어 반지름)
        OUTER_RADIUS    float   그레인 전체 외부 반지름 [m]
        RHO             float   연료 밀도 [kg/m³]
        A               float   후퇴율 계수 a  (rdot = a * G^n)
        N               float   후퇴율 지수 n

        선택 키 (기본값 있음)
        ─────────────────────────────────────────────────────
        GRAIN_TYPE               str     그레인 형상: "PORT"(원형 다공 포트, 기본) |
                                          "FIN"(멀티핀/wagon-wheel, 직사각형 슬롯) |
                                          "FINOCYL"(피노실, 슬롯 끝 반원형 릴리프)
        PORT_NUM                 int     PORT 타입일 때 포트 개수 (기본 1)
        PORT_CENTER_RADIUS       float   PORT 타입 다중 포트 중심 배치 반지름 [m]
        FIN_NUM                  int     FIN/FINOCYL 타입일 때 핀(슬롯) 개수 (기본 6)
        FIN_LENGTH                float   FIN/FINOCYL 타입 핀의 반경 방향 길이 [m]
                                          (기본 (OUTER_RADIUS - INNER_RADIUS) * 0.6)
        FIN_WIDTH                 float   FIN/FINOCYL 타입 핀 밑동(보어 쪽) 폭 [m]
                                          (기본 INNER_RADIUS * 0.6)
        FIN_TIP_WIDTH             float   FIN/FINOCYL 타입 핀 끝단(바깥쪽) 폭 [m]
                                          (기본 FIN_WIDTH, 즉 테이퍼 없음)
                                          FIN_WIDTH보다 작게 주면 밑동은 두껍고 끝은
                                          얇아지는 star-point 형태가 되어, 점화 직후
                                          뾰족한 끝이 먼저 타버리며 연소면적이 급격히
                                          줄어드는(초반 피크 후 급락) 곡선을 재현한다.
                                          0을 주면 완전히 뾰족한 쐐기(triangle) 핀.
        GRID_SIZE                int     레벨셋 격자 해상도 (기본 300)
        GRAIN_SNAPSHOT_INTERVAL float   스냅샷 캡처 간격 [s] (기본 1.0)
                                        웹 시각화용 그레인 단면 캡처 주기
    """

    # ──────────────────────────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, config: dict):
        self.name         = config["NAME"]
        self.grain_type   = str(config.get("GRAIN_TYPE", "PORT")).upper()
        self.port_num     = config.get("PORT_NUM", 1)
        self.length       = config["LENGTH"]
        self.inner_radius = config["INNER_RADIUS"]
        self.outer_radius = config["OUTER_RADIUS"]
        self.rho          = config["RHO"]
        self.a            = config["A"]
        self.n            = config["N"]

        self.grid_size = config.get("GRID_SIZE", 300)

        default_cr = self.outer_radius * 0.5 if self.port_num > 1 else 0.0
        self.port_center_radius = config.get("PORT_CENTER_RADIUS", default_cr)

        # ── 멀티핀(wagon-wheel) 그레인 옵션 ──────────────────────────────
        self.fin_num    = config.get("FIN_NUM", 6)
        self.fin_length = config.get(
            "FIN_LENGTH", (self.outer_radius - self.inner_radius) * 0.6
        )
        self.fin_width     = config.get("FIN_WIDTH", self.inner_radius * 0.6)
        self.fin_tip_width = config.get("FIN_TIP_WIDTH", self.fin_width)

        # ── 웹 시각화용 스냅샷 수집 설정 ────────────────────────────────
        self._elapsed_time      = 0.0
        self._grain_snapshots   = []                              # 수집된 스냅샷 목록
        snap_iv = float(config.get("GRAIN_SNAPSHOT_INTERVAL", 0.1))
        self._snapshot_interval = snap_iv
        self._next_snapshot_time = 0.0   # t=0 에서 즉시 첫 번째 캡처

        self._init_levelset()

    # ──────────────────────────────────────────────────────────────────────
    # 레벨셋 초기화
    # ──────────────────────────────────────────────────────────────────────

    def _init_levelset(self):
        """
        포트 개수·반지름·외부 반지름을 기반으로 레벨셋 phi 배열을 초기화합니다.

        phi > 0 : 고체 연료 영역
        phi < 0 : 가스(포트) 영역
        phi = 0 : 연소 표면
        """
        gs  = self.grid_size
        ext = self.outer_radius

        x = np.linspace(-ext, ext, gs)
        y = np.linspace(-ext, ext, gs)
        self.X, self.Y = np.meshgrid(x, y)
        self.dx = x[1] - x[0]
        self.dy = self.dx

        if self.grain_type == "FIN":
            solid_mask, port_centers = self._build_fin_mask()
        elif self.grain_type == "FINOCYL":
            solid_mask, port_centers = self._build_finocyl_mask()
        else:
            solid_mask, port_centers = self._build_port_mask()
        self._port_centers = port_centers

        dist_solid = distance_transform_edt(solid_mask)
        dist_gas   = distance_transform_edt(~solid_mask)
        self.phi   = (dist_solid - dist_gas) * self.dx

        # ── t=0 초기 그레인 형상 캡처 ────────────────────────────────────
        self._capture_snapshot(0.0)
        self._next_snapshot_time = self._snapshot_interval

    def _build_port_mask(self):
        """원형 다공 포트(circular multi-port) 고체 마스크를 생성합니다."""
        gs = self.grid_size

        if self.port_num == 1:
            port_centers = [(0.0, 0.0)]
        else:
            r_c = self.port_center_radius
            port_centers = [
                (r_c * np.cos(2 * np.pi * k / self.port_num),
                 r_c * np.sin(2 * np.pi * k / self.port_num))
                for k in range(self.port_num)
            ]

        solid_mask = np.ones((gs, gs), dtype=bool)
        for cx, cy in port_centers:
            r = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            solid_mask[r <= self.inner_radius] = False

        return solid_mask, port_centers

    def _build_fin_mask(self):
        """
        멀티핀(wagon-wheel) 그레인 고체 마스크를 생성합니다.

        중앙 원형 보어(반지름 INNER_RADIUS)에서 FIN_NUM개의 핀 슬롯이 반경 방향으로
        FIN_LENGTH만큼 뻗어나가는 형상입니다. 슬롯 폭은 밑동(FIN_WIDTH)에서
        끝단(FIN_TIP_WIDTH)까지 선형으로 테이퍼됩니다. FIN_TIP_WIDTH가 FIN_WIDTH와
        같으면(기본값) 기존과 동일한 직사각형 슬롯이 되고, 0에 가까울수록 끝이
        뾰족한 star-point 형태가 됩니다. 각 슬롯은 보어 중심 기준 등각도로 배치됩니다.
        """
        solid_mask = np.ones_like(self.X, dtype=bool)

        r = np.sqrt(self.X**2 + self.Y**2)
        solid_mask[r <= self.inner_radius] = False

        half_root = self.fin_width / 2
        half_tip  = self.fin_tip_width / 2
        fin_len   = max(self.fin_length, 1e-12)

        for k in range(self.fin_num):
            ang = 2 * np.pi * k / self.fin_num
            # 핀 축이 로컬 x축과 일치하도록 좌표 회전
            xr =  self.X * np.cos(ang) + self.Y * np.sin(ang)
            yr = -self.X * np.sin(ang) + self.Y * np.cos(ang)

            # 밑동(half_root) → 끝단(half_tip) 선형 테이퍼
            frac    = np.clip((xr - self.inner_radius) / fin_len, 0.0, 1.0)
            half_w  = half_root + (half_tip - half_root) * frac

            in_fin = (
                (xr >= self.inner_radius) &
                (xr <= self.inner_radius + self.fin_length) &
                (np.abs(yr) <= half_w)
            )
            solid_mask[in_fin] = False

        return solid_mask, [(0.0, 0.0)]

    def _build_finocyl_mask(self):
        """
        피노실(finocyl) 그레인 고체 마스크를 생성합니다.

        중앙 원형 보어(반지름 INNER_RADIUS)에서 FIN_NUM개의 핀 슬롯이 뻗어나가는
        점은 FIN 타입과 동일하게 밑동(FIN_WIDTH)→끝단(FIN_TIP_WIDTH) 테이퍼를
        따르되, 슬롯 끝을 반지름 FIN_TIP_WIDTH/2인 반원으로 마감하여(응력 완화용
        릴리프 슬롯) 실제 피노실 그레인의 특징인 "둥근 팁"을 반영합니다.
        FIN_TIP_WIDTH가 0이면 반원 캡이 점으로 축소되어 뾰족한 핀이 됩니다.
        """
        solid_mask = np.ones_like(self.X, dtype=bool)

        r = np.sqrt(self.X**2 + self.Y**2)
        solid_mask[r <= self.inner_radius] = False

        half_root = self.fin_width / 2
        half_tip  = self.fin_tip_width / 2
        straight_len = max(self.fin_length - half_tip, 0.0)
        rect_end  = self.inner_radius + straight_len

        for k in range(self.fin_num):
            ang = 2 * np.pi * k / self.fin_num
            xr =  self.X * np.cos(ang) + self.Y * np.sin(ang)
            yr = -self.X * np.sin(ang) + self.Y * np.cos(ang)

            # 밑동(half_root) → 직선 구간 끝(half_tip) 선형 테이퍼
            frac   = np.clip((xr - self.inner_radius) / max(straight_len, 1e-12), 0.0, 1.0)
            half_w = half_root + (half_tip - half_root) * frac

            in_rect = (
                (xr >= self.inner_radius) &
                (xr <= rect_end) &
                (np.abs(yr) <= half_w)
            )
            in_cap = (
                (xr >= rect_end) &
                ((xr - rect_end)**2 + yr**2 <= half_tip**2)
            )
            solid_mask[in_rect | in_cap] = False

        return solid_mask, [(0.0, 0.0)]

    # ──────────────────────────────────────────────────────────────────────
    # 스냅샷 수집 (내부)
    # ──────────────────────────────────────────────────────────────────────

    def _capture_snapshot(self, t: float):
        """
        현재 phi=0 등고선을 물리 좌표로 변환하고, 포트 면적·둘레와 함께
        웹 전송용 JSON 직렬화 가능 딕셔너리로 저장합니다.

        contour 좌표 변환:
            픽셀 인덱스 (row, col) → 물리 좌표 (x [m], y [m])
            x = -ext + col * dx
            y = -ext + row * dy
        각 등고선은 최대 200 포인트로 다운샘플링하여 전송 크기를 제한합니다.
        """
        try:
            from skimage.measure import find_contours
        except ImportError:
            return

        contours_px = find_contours(self.phi, level=0.0)
        ext = self.outer_radius

        perimeter      = 0.0
        contours_phys  = []

        for c in contours_px:
            # ── 둘레 계산 (픽셀 공간에서 물리 단위로) ──────────────────
            drow = np.diff(c[:, 0]) * self.dy
            dcol = np.diff(c[:, 1]) * self.dx
            perimeter += float(np.sum(np.hypot(drow, dcol)))

            # ── 물리 좌표 변환 ──────────────────────────────────────────
            x_phys = -ext + c[:, 1] * self.dx
            y_phys = -ext + c[:, 0] * self.dy
            pts    = np.column_stack([x_phys, y_phys])

            # ── 다운샘플링: 최대 200 포인트 ─────────────────────────────
            step = max(1, len(pts) // 200)
            contours_phys.append(pts[::step].tolist())

        port_area = float(np.sum(self.phi < 0) * self.dx * self.dy)

        self._grain_snapshots.append({
            't':          round(t, 4),
            'contours':   contours_phys,
            'port_area':  round(port_area, 8),   # [m²]
            'perimeter':  round(perimeter, 6),   # [m]
        })

    # ──────────────────────────────────────────────────────────────────────
    # 스냅샷 공개 API
    # ──────────────────────────────────────────────────────────────────────

    def get_snapshots(self) -> list:
        """
        시뮬레이션 중 캡처된 그레인 단면 스냅샷 목록을 반환합니다.

        Returns
        -------
        list of dict:
            [
              {
                't':         float,   # 시뮬레이션 시각 [s]
                'contours':  [        # phi=0 등고선 목록
                    [[x, y], ...],    # 각 등고선: [m] 단위 좌표 목록
                ],
                'port_area': float,   # 현재 포트 단면적 [m²]
                'perimeter': float,   # 연소 표면 둘레 [m]
              },
              ...
            ]
        """
        return self._grain_snapshots

    def get_grain_meta(self) -> dict:
        """
        웹 시각화에 필요한 메타데이터를 반환합니다.

        Returns
        -------
        dict:
            'extent':       float          격자 범위 (= outer_radius) [m]
            'port_centers': [[x,y], ...]   포트 중심 좌표 목록 [m]
            'grain_type':   str            "PORT" | "FIN"
        """
        return {
            'extent':       float(self.outer_radius),
            'port_centers': [[float(cx), float(cy)] for cx, cy in self._port_centers],
            'grain_type':   self.grain_type,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 레벨셋 수치 기법 (정적 내부 메서드)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _grad_godunov(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Godunov 풍상차분법으로 |∇φ|⁺ 계산. 논문 식 (9)."""
        Dmx = (phi - np.roll(phi,  1, axis=1)) / dx
        Dpx = (np.roll(phi, -1, axis=1) - phi) / dx
        Dmy = (phi - np.roll(phi,  1, axis=0)) / dy
        Dpy = (np.roll(phi, -1, axis=0) - phi) / dy

        Dmx[:, 0]  = Dmx[:, 1];    Dpx[:, -1] = Dpx[:, -2]
        Dmy[0, :]  = Dmy[1, :];    Dpy[-1, :] = Dpy[-2, :]

        return np.sqrt(
            np.maximum(Dmx, 0)**2 + np.minimum(Dpx, 0)**2 +
            np.maximum(Dmy, 0)**2 + np.minimum(Dpy, 0)**2
        )

    @classmethod
    def _tvd_rk3(cls, phi: np.ndarray, F: float,
                 dt: float, dx: float, dy: float) -> np.ndarray:
        """3차 TVD Runge-Kutta 시간 전진법. 논문 식 (11)~(14)."""
        def L(p):
            return -F * cls._grad_godunov(p, dx, dy)

        phi1 = phi + dt * L(phi)
        phi2 = 0.75 * phi + 0.25 * phi1 + 0.25 * dt * L(phi1)
        return (1/3) * phi + (2/3) * phi2 + (2/3) * dt * L(phi2)

    # ──────────────────────────────────────────────────────────────────────
    # 핵심 API: 레벨셋 전진 + 연소 표면 선적분
    # ──────────────────────────────────────────────────────────────────────

    def advance_and_measure(self, rdot: float, dt: float) -> float:
        """
        레벨셋을 rdot 속도로 dt 시간만큼 전진시킨 뒤,
        phi=0 등고선의 둘레(선적분 값) [m]를 반환합니다.
        """
        if rdot <= 0:
            return self._compute_perimeter()

        dt_cfl = 0.5 * min(self.dx, self.dy) / rdot
        n_sub  = max(1, math.ceil(dt / dt_cfl))
        dt_sub = dt / n_sub

        phi = self.phi
        for _ in range(n_sub):
            phi = self._tvd_rk3(phi, rdot, dt_sub, self.dx, self.dy)
        self.phi = phi

        return self._compute_perimeter()

    def _compute_perimeter(self) -> float:
        """현재 phi 배열에서 phi=0 등고선의 둘레 [m]를 선적분합니다."""
        try:
            from skimage.measure import find_contours
        except ImportError as e:
            raise ImportError("pip install scikit-image") from e

        contours  = find_contours(self.phi, level=0.0)
        perimeter = 0.0
        for c in contours:
            drow = np.diff(c[:, 0]) * self.dy
            dcol = np.diff(c[:, 1]) * self.dx
            perimeter += np.sum(np.hypot(drow, dcol))
        return perimeter

    # ──────────────────────────────────────────────────────────────────────
    # 기존 인터페이스 유지 (burn, rdot_G)
    # ──────────────────────────────────────────────────────────────────────

    def rdot_G(self, G: float) -> float:
        """산화제 질량 유속 G [kg/(m²·s)]로부터 후퇴율 [m/s] 계산."""
        return self.a * G ** self.n

    def burn(self, mox_dot: float, dt: float):
        """
        dt 시간 동안 연소를 시뮬레이션합니다. (기존 인터페이스 호환)

        내부적으로 레벨셋을 전진시키고, snapshot_interval마다 그레인
        단면 형상을 캡처하여 _grain_snapshots에 축적합니다.

        Parameters
        ----------
        mox_dot : float  산화제 질량 유량 [kg/s]
        dt      : float  시간 스텝 [s]

        Returns
        -------
        burn_area : float  연소 면적 [m²]
        dm        : float  연료 질량 유량 [kg/s]
        rdot      : float  연소 후퇴율 [m/s]
        """
        port_area = np.sum(self.phi < 0) * self.dx * self.dy
        if port_area <= 0:
            return 0.0, 0.0, 0.0

        G    = mox_dot / port_area
        rdot = self.rdot_G(G)

        perimeter = self.advance_and_measure(rdot, dt)
        burn_area = perimeter * self.length
        dm        = rdot * self.rho * burn_area

        new_port_area = np.sum(self.phi < 0) * self.dx * self.dy
        if self.grain_type in ("FIN", "FINOCYL"):
            # FIN/FINOCYL 타입은 보어+핀이 합쳐진 형상이라 등가 단일 원형 반지름으로 근사
            self.inner_radius = math.sqrt(max(new_port_area, 0.0) / math.pi)
        else:
            self.inner_radius = math.sqrt(
                max(new_port_area, 0.0) / (math.pi * self.port_num)
            )

        # ── 시간 추적 및 스냅샷 캡처 ─────────────────────────────────────
        self._elapsed_time += dt
        if self._elapsed_time >= self._next_snapshot_time:
            self._capture_snapshot(self._elapsed_time)
            self._next_snapshot_time += self._snapshot_interval

        return burn_area, dm, rdot

    # ──────────────────────────────────────────────────────────────────────
    # 시각화 유틸리티 (matplotlib, 로컬 디버깅용)
    # ──────────────────────────────────────────────────────────────────────

    def plot_current_grain(self, ax=None, color="royalblue", label=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.contour(self.X, self.Y, self.phi, levels=[0], colors=[color], linewidths=1.5)
        ax.plot([], [], color=color, linewidth=1.5, label=label or "Burn surface")
        return ax

    def __str__(self):
        perim = self._compute_perimeter()
        snaps = len(self._grain_snapshots)
        return (
            f"Fuel '{self.name}' | 포트 {self.port_num}개 | "
            f"내부 반지름(등가): {self.inner_radius*1000:.2f} mm | "
            f"외부 반지름: {self.outer_radius*1000:.1f} mm | "
            f"연소 둘레: {perim*1000:.2f} mm | "
            f"스냅샷: {snaps}개"
        )
