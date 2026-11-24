"""
PIPEDB.csv 관리 유틸리티
────────────────────────────────────────────────────────────────────────────
시중에서 구매 가능한 알루미늄 6061 파이프 정보를 PIPEDB.csv 에 등록/삭제/조회한다.
모든 수치는 SI 단위 기준이다. (길이 [m], 응력 [Pa], 밀도 [kg/m^3])

컬럼 구조
    pipe_id           파이프 고유 ID (중복 불가)
    name              표시용 이름 / 규격 설명
    material          재질 (예: AL6061)
    temper            열처리 (예: T6)
    outer_diameter    구매 파이프 외경 [m]
    inner_diameter    구매 파이프 내경 [m]  ← 가공하지 않고 그대로 사용 (고정)
    wall_thickness    구매 파이프 두께 [m]  = (외경 - 내경) / 2 (자동 계산)
    length            구매 파이프 길이 [m]
    yield_strength    항복 강도 [Pa]
    tensile_strength  인장 강도 [Pa]
    density           재질 밀도 [kg/m^3]
    price             가격 (통화 자유, 참고용)
    supplier          구매처
    note              비고

CLI 사용 예
    python managepipedb.py list
    python managepipedb.py show AL6061-OD110-T5-L1500
    python managepipedb.py add --pipe-id AL6061-OD110-T5-L1000 \\
        --outer-diameter 0.11 --inner-diameter 0.1 --length 1.0 --price 70000
    python managepipedb.py delete AL6061-OD110-T5-L1000

다른 모듈에서 사용
    from managepipedb import load_pipes, get_pipe
"""

import argparse
import csv
import os
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PIPEDB.csv")

COLUMNS = [
    "pipe_id", "name", "material", "temper",
    "outer_diameter", "inner_diameter", "wall_thickness", "length",
    "yield_strength", "tensile_strength", "density",
    "price", "supplier", "note",
]

FLOAT_COLUMNS = [
    "outer_diameter", "inner_diameter", "wall_thickness", "length",
    "yield_strength", "tensile_strength", "density", "price",
]

# Al 6061-T6 기본 물성 (add 시 값을 주지 않으면 사용)
DEFAULTS = {
    "material": "AL6061",
    "temper": "T6",
    "yield_strength": 276e6,
    "tensile_strength": 310e6,
    "density": 2700.0,
    "price": 0.0,
    "supplier": "",
    "note": "",
}

# 두께 자동 계산값과 입력값의 허용 오차 [m]
THICKNESS_TOL = 1e-6


# ──────────────────────────────────────────────────────────────────────────
# 읽기 / 쓰기
# ──────────────────────────────────────────────────────────────────────────

def _parse_row(row):
    pipe = dict(row)
    for col in FLOAT_COLUMNS:
        pipe[col] = float(pipe[col]) if pipe.get(col) not in (None, "") else 0.0
    return pipe


def load_pipes(path=DB_PATH):
    """PIPEDB.csv 를 읽어 파이프 dict 리스트로 반환한다 (수치 컬럼은 float)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [_parse_row(row) for row in csv.DictReader(f) if row.get("pipe_id")]


def get_pipe(pipe_id, path=DB_PATH):
    """pipe_id 에 해당하는 파이프 dict 를 반환한다. 없으면 None."""
    for pipe in load_pipes(path):
        if pipe["pipe_id"] == pipe_id:
            return pipe
    return None


def save_pipes(pipes, path=DB_PATH):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for pipe in pipes:
            writer.writerow({col: _fmt(pipe.get(col, "")) for col in COLUMNS})


def _fmt(value):
    if isinstance(value, float):
        return repr(int(value)) if value.is_integer() and abs(value) >= 1e3 else repr(value)
    return value


# ──────────────────────────────────────────────────────────────────────────
# 추가 / 삭제
# ──────────────────────────────────────────────────────────────────────────

def validate_pipe(pipe):
    """파이프 데이터 검증. 문제가 있으면 ValueError."""
    if not pipe.get("pipe_id"):
        raise ValueError("pipe_id 가 비어 있습니다.")
    if "," in pipe["pipe_id"]:
        raise ValueError("pipe_id 에 ',' 를 사용할 수 없습니다.")

    od, idia, length = pipe["outer_diameter"], pipe["inner_diameter"], pipe["length"]
    if od <= 0 or idia <= 0 or length <= 0:
        raise ValueError("외경, 내경, 길이는 양수여야 합니다. (SI 단위 [m])")
    if od <= idia:
        raise ValueError(f"외경({od}) 이 내경({idia}) 보다 커야 합니다.")
    if od > 1.0:
        raise ValueError(f"외경 {od} m 는 비정상적으로 큽니다. 단위를 [m] 로 입력했는지 확인하세요.")

    t_calc = (od - idia) / 2
    if abs(pipe["wall_thickness"] - t_calc) > THICKNESS_TOL:
        raise ValueError(
            f"두께({pipe['wall_thickness']}) 가 (외경-내경)/2 = {t_calc} 와 다릅니다."
        )
    for col in ("yield_strength", "tensile_strength", "density"):
        if pipe[col] <= 0:
            raise ValueError(f"{col} 은 양수여야 합니다.")


def add_pipe(pipe_id, outer_diameter, inner_diameter, length, name=None,
             wall_thickness=None, overwrite=False, path=DB_PATH, **kwargs):
    """
    파이프를 DB 에 추가한다. wall_thickness 를 생략하면 (외경-내경)/2 로 자동 계산.
    같은 pipe_id 가 있으면 overwrite=True 일 때만 덮어쓴다.
    """
    pipe = dict(DEFAULTS)
    pipe.update({k: v for k, v in kwargs.items() if v is not None and k in COLUMNS})
    pipe.update({
        "pipe_id": pipe_id,
        "name": name if name is not None else pipe_id,
        "outer_diameter": float(outer_diameter),
        "inner_diameter": float(inner_diameter),
        "length": float(length),
    })
    pipe["wall_thickness"] = (
        float(wall_thickness) if wall_thickness is not None
        else round((pipe["outer_diameter"] - pipe["inner_diameter"]) / 2, 9)
    )
    for col in FLOAT_COLUMNS:
        pipe[col] = float(pipe[col])

    validate_pipe(pipe)

    pipes = load_pipes(path)
    idx = next((i for i, p in enumerate(pipes) if p["pipe_id"] == pipe_id), None)
    if idx is not None:
        if not overwrite:
            raise ValueError(f"pipe_id '{pipe_id}' 가 이미 존재합니다. (덮어쓰려면 overwrite)")
        pipes[idx] = pipe
    else:
        pipes.append(pipe)

    save_pipes(pipes, path)
    return pipe


def delete_pipe(pipe_id, path=DB_PATH):
    """pipe_id 파이프를 삭제한다. 삭제했으면 True, 없으면 False."""
    pipes = load_pipes(path)
    remaining = [p for p in pipes if p["pipe_id"] != pipe_id]
    if len(remaining) == len(pipes):
        return False
    save_pipes(remaining, path)
    return True


# ──────────────────────────────────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────────────────────────────────

def print_pipes(pipes):
    if not pipes:
        print("등록된 파이프가 없습니다.")
        return
    header = f"{'pipe_id':<28} {'OD[mm]':>8} {'ID[mm]':>8} {'t[mm]':>7} {'L[mm]':>8} {'Sy[MPa]':>8} {'price':>10}"
    print(header)
    print("-" * len(header))
    for p in pipes:
        print(
            f"{p['pipe_id']:<28} {p['outer_diameter']*1e3:>8.2f} {p['inner_diameter']*1e3:>8.2f} "
            f"{p['wall_thickness']*1e3:>7.2f} {p['length']*1e3:>8.1f} "
            f"{p['yield_strength']/1e6:>8.1f} {p['price']:>10.0f}"
        )
    print(f"\n총 {len(pipes)} 개")


def print_pipe_detail(pipe):
    for col in COLUMNS:
        print(f"  {col:<17}: {pipe[col]}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="PIPEDB.csv 관리 (모든 단위 SI)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="등록된 파이프 목록")

    p_show = sub.add_parser("show", help="파이프 상세 정보")
    p_show.add_argument("pipe_id")

    p_add = sub.add_parser("add", help="파이프 추가")
    p_add.add_argument("--pipe-id", required=True)
    p_add.add_argument("--name")
    p_add.add_argument("--outer-diameter", type=float, required=True, help="외경 [m]")
    p_add.add_argument("--inner-diameter", type=float, required=True, help="내경 [m]")
    p_add.add_argument("--length", type=float, required=True, help="길이 [m]")
    p_add.add_argument("--material")
    p_add.add_argument("--temper")
    p_add.add_argument("--yield-strength", type=float, help="항복 강도 [Pa] (기본 276e6)")
    p_add.add_argument("--tensile-strength", type=float, help="인장 강도 [Pa] (기본 310e6)")
    p_add.add_argument("--density", type=float, help="밀도 [kg/m^3] (기본 2700)")
    p_add.add_argument("--price", type=float)
    p_add.add_argument("--supplier")
    p_add.add_argument("--note")
    p_add.add_argument("--overwrite", action="store_true", help="같은 ID 가 있으면 덮어쓰기")

    p_del = sub.add_parser("delete", help="파이프 삭제")
    p_del.add_argument("pipe_id")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "list":
            print_pipes(load_pipes())

        elif args.cmd == "show":
            pipe = get_pipe(args.pipe_id)
            if pipe is None:
                print(f"'{args.pipe_id}' 를 찾을 수 없습니다.")
                return 1
            print_pipe_detail(pipe)

        elif args.cmd == "add":
            pipe = add_pipe(
                pipe_id=args.pipe_id,
                name=args.name,
                outer_diameter=args.outer_diameter,
                inner_diameter=args.inner_diameter,
                length=args.length,
                material=args.material,
                temper=args.temper,
                yield_strength=args.yield_strength,
                tensile_strength=args.tensile_strength,
                density=args.density,
                price=args.price,
                supplier=args.supplier,
                note=args.note,
                overwrite=args.overwrite,
            )
            print(f"추가 완료: {pipe['pipe_id']}")
            print_pipe_detail(pipe)

        elif args.cmd == "delete":
            if delete_pipe(args.pipe_id):
                print(f"삭제 완료: {args.pipe_id}")
            else:
                print(f"'{args.pipe_id}' 를 찾을 수 없습니다.")
                return 1

    except ValueError as e:
        print(f"[오류] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
