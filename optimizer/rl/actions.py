from collections import Counter

ACTIONS = [
    {"name": "기본",       "tournament": 3, "crossover": 0.9, "alpha": 0.3, "rate": 0.15, "sigma": 0.10},
    {"name": "넓은 탐색",  "tournament": 2, "crossover": 0.9, "alpha": 0.5, "rate": 0.30, "sigma": 0.20},
    {"name": "집중 활용",  "tournament": 5, "crossover": 0.9, "alpha": 0.0, "rate": 0.10, "sigma": 0.03},
    {"name": "국소 탐색",  "tournament": 5, "crossover": 0.5, "alpha": 0.1, "rate": 0.20, "sigma": 0.02},
    {"name": "다양성 주입", "tournament": 3, "crossover": 0.9, "alpha": 0.3, "rate": 0.15, "sigma": 0.10, "inject": 0.25},
    {"name": "미세 조정",  "tournament": 3, "crossover": 0.0, "alpha": 0.0, "rate": 0.50, "sigma": 0.01},
    {"name": "교배 위주",  "tournament": 3, "crossover": 1.0, "alpha": 0.3, "rate": 0.05, "sigma": 0.05},
    {"name": "제약 수리",  "tournament": 3, "crossover": 0.0, "alpha": 0.0, "rate": 0.50, "sigma": 0.10, "repair": True},
]


N_ACTIONS = len(ACTIONS)

# 제약 이름 → 그 제약에 영향을 주는 유전자
REPAIR_GENES = {
    "Impulse":          ["oxidizer_M", "fuel_length", "outter_radius", "inner_radius", "port_num"],
    "ThrustAvg":        ["injector_oriffice_n", "injector_oriffice_radius", "nozzle_throat_radius",
                         "inner_radius", "port_num"],
    "Tcc":              ["oxidizer_tank_tickness", "nozzle_throat_radius", "injector_oriffice_radius"],
    "Toxt":             ["oxidizer_tank_tickness", "oxidizer_M", "oxidizer_tank_length"],
    "OxtFill":          ["oxidizer_M", "oxidizer_tank_length"],
    "OxtVolume":        ["oxidizer_M", "oxidizer_tank_length"],
    "SimCompleted":     ["nozzle_throat_radius", "injector_oriffice_radius", "oxidizer_M"],
    "ChamberThickness": ["oxidizer_tank_tickness"],
    "ChamberLength":    ["oxidizer_tank_length", "combution_chamber_length"],
    "InjectorArea":     ["injector_oriffice_n", "injector_oriffice_radius"],
    "MaxOri":           ["injector_oriffice_length"],
    "FuelArea":         ["inner_radius", "port_num", "outter_radius"],
    "FuelPort":         ["port_center_radius", "inner_radius", "outter_radius"],
    "PortWeb":          ["port_center_radius", "inner_radius", "port_num"],
    "PortResolution":   ["inner_radius", "outter_radius"],
    "FuelLenght":       ["fuel_length", "outter_radius", "combution_chamber_length"],
    "FuelRadius":       ["outter_radius"],
    "NozzleThroatSize": ["nozzle_throat_radius"],
    "NozzleExpansion":  ["nozzle_exit_radius", "nozzle_throat_radius"],
    "NozzleExitSize":   ["nozzle_exit_radius", "oxidizer_tank_tickness"],
    # 인젝터 면적을 줄이거나 노즐 목을 키우면 챔버압이 내려가 (Pt - Pc)/Pc 가 커진다
    "InjectorDP":       ["injector_oriffice_n", "injector_oriffice_radius", "nozzle_throat_radius"],
}

def resolve(action_id, evaluations):
    params = dict(ACTIONS[action_id])
    if params.pop("repair", False):
        counts = Counter(name for ev in evaluations for name in ev["failed"])
        if counts:
            top = counts.most_common(1)[0][0]
            params["genes"] = REPAIR_GENES.get(top)
    return params
