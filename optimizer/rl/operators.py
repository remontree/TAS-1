"""
전략 설정값(params)을 인자로 받는 GA 연산자
GA.py 의 tournament / crossover / mutate 와 같지만, 전역 상수 대신 인자를 쓴다.
"""

import GA


def tournament(rng, evaluations, size):
    contenders = rng.sample(evaluations, min(size, len(evaluations)))
    return min(contenders, key=GA.rank_key)["genome"]


def crossover(rng, a, b, rate, alpha):
    if rng.random() > rate:
        return dict(a), dict(b)
    c1, c2 = {}, {}
    c1["pipe"], c2["pipe"] = (a["pipe"], b["pipe"]) if rng.random() < 0.5 else (b["pipe"], a["pipe"])
    for var in GA.DESIGN_VARS:
        lo, hi = min(a[var], b[var]), max(a[var], b[var])
        span = hi - lo
        c1[var] = rng.uniform(lo - alpha * span, hi + alpha * span)
        c2[var] = rng.uniform(lo - alpha * span, hi + alpha * span)
    return c1, c2


def mutate(rng, g, n_pipes, rate, sigma, genes=None):
    """genes: 변이할 유전자 이름 목록 (None = 전체, 이때만 파이프도 변이)"""
    g = dict(g)
    if genes is None:
        genes = GA.DESIGN_VARS
        if n_pipes > 1 and rng.random() < rate:
            g["pipe"] = rng.randrange(n_pipes)
    for var in genes:
        if rng.random() >= rate:
            continue
        lo, hi = GA.BOUNDS[var]
        if var in GA.INT_VARS:
            step = max(1, round(sigma * (hi - lo)))
            g[var] = g[var] + rng.randint(-step, step)
        else:
            g[var] = g[var] + rng.gauss(0.0, sigma * (hi - lo))
    return GA.quantize(g)


def make_next_population(rng, evaluations, pipes, params, pop_size, elite_count=GA.ELITE_COUNT):
    """evaluations 는 rank_key 로 정렬돼 있어야 한다."""
    next_pop = [ev["genome"] for ev in evaluations[:elite_count]]
    n_children = pop_size - int(round(params.get("inject", 0.0) * pop_size))
    genes = params.get("genes")
    while len(next_pop) < n_children:
        a = tournament(rng, evaluations, params["tournament"])
        b = tournament(rng, evaluations, params["tournament"])
        c1, c2 = crossover(rng, a, b, params["crossover"], params["alpha"])
        next_pop.append(mutate(rng, c1, len(pipes), params["rate"], params["sigma"], genes))
        if len(next_pop) < n_children:
            next_pop.append(mutate(rng, c2, len(pipes), params["rate"], params["sigma"], genes))
    while len(next_pop) < pop_size:
        try:
            next_pop.append(GA.sample_genome(rng, pipes))
        except ValueError:
            continue
    return next_pop
