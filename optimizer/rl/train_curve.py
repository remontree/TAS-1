"""
곡선 판단 에이전트 (rl/curve_agent.py) 학습
실행 (optimizer 폴더에서)
    python -m rl.train_curve --steps 6000000 [--ts-model rl/ts_surrogate.pt]
모니터링: http://127.0.0.1:10000/training  (kind=curve)

단계
  1. baseline : 같은 대리모델 위 기본 GA (32 × 100세대) — 대리모델 평가 기준 + core 검증 기준
  2. teacher  : 한 걸음 앞 보기 — 에이전트 상태마다 이동 115가지를 모두 대리모델로 해 보고 점수 변화를 기록
                (행동: 앞 보기 최선 2개, 확률 eps 로 무작위 → 다양한 상태)
  3. pretrain : 정책 로짓 ← softmax(점수 변화 / tau) 증류 (곡선을 보고 어떤 변수를 얼마나 바꿀지의 첫 판단력)
  4. running  : PPO 미세 조정 (여러 걸음 계획 — 잠시 손해 보는 이동, 큰 이동 뒤 미세 조정 ...)
                eval_freq 마다 대리모델 평가, core_freq 마다 최종 설계를 진짜 core 로 검증 → core 점수로 best_model 선택
결과: rl/runs/curve_<시각>/  best_model.zip (core 기준), best_sur_model.zip (대리모델 기준), final_model.zip
"""

import argparse
import csv
import json
import multiprocessing
import os
import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecNormalize

import GA
from rl import curve_agent as ca
from rl import direct, verify_core
from rl.train_direct import EVAL_POP, agent_eval, baseline_eval, eval_episodes, summarize
from rl.train_ppo import LiveWriter, Status

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
VAR_KEYS = GA.DESIGN_VARS + ["pipe", "none"]


def _var_key(a):
    var, _ = ca.move_label(a)
    if var is None:
        return "none"
    return "pipe" if var.startswith("pipe") else var


def write_live(live, row):
    live.f.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}) + "\n")
    live.f.flush()


# ──────────────────────────────────────────────────────────────────────────
# 2. 한 걸음 앞 보기 데이터
# ──────────────────────────────────────────────────────────────────────────

def collect_teacher(env, n_states, eps, live, status, log=print):
    """(obs (N, OBS_DIM), gain (N, N_MOVES)) — gain = 이동 하나를 했을 때의 점수 변화"""
    obs = env.reset()
    O, G = [], []
    n_iter = int(np.ceil(n_states / env.n))
    t0 = time.time()
    rng = np.random.default_rng(0)
    finals = deque(maxlen=2000)
    for it in range(n_iter):
        gain = env.lookahead(list(range(env.n)))
        O.append(obs.copy())
        G.append(gain.astype(np.float32))
        order = np.argsort(-gain, 1)
        a1, a2 = order[:, 0], order[:, 1]
        a2 = np.where(gain[np.arange(env.n), a2] > 0, a2, ca.NONE)
        act = np.stack([a1, a2], 1)
        rand = rng.random(act.shape) < eps
        act[rand] = rng.integers(0, ca.N_MOVES, rand.sum())
        obs, _, _, infos = env.step(act)
        for info in infos:
            if "best_isp" in info:
                finals.append(info)
        if it % 10 == 9 or it == n_iter - 1:
            el = time.time() - t0
            fin = list(finals)
            row = {"step": (it + 1) * env.n, "t": round(time.time(), 1),
                   "teacher/states": (it + 1) * env.n,
                   "teacher/best_gain": float(np.mean(gain.max(1))),
                   "teacher/positive": float(np.mean(gain.max(1) > 1e-6)),
                   "teacher/success": float(np.mean([f["feasible"] for f in fin])) if fin else None,
                   "teacher/best_isp": float(np.mean([f["best_isp"] for f in fin if f["feasible"]])) if fin and any(f["feasible"] for f in fin) else None}
            write_live(live, row)
            status.update(phase_step=(it + 1) * env.n, phase_total=n_iter * env.n,
                          phase_eta=el / (it + 1) * (n_iter - it - 1))
            status.save()
            log(f"  [teacher] {(it + 1) * env.n:,}/{n_iter * env.n:,} 상태 · 최선 이동 이득 {row['teacher/best_gain']:.3f} "
                f"· 개선 가능 {row['teacher/positive']:.0%} · {el:.0f}s", flush=True)
    return np.concatenate(O), np.concatenate(G)


# ──────────────────────────────────────────────────────────────────────────
# 3. 증류 사전학습
# ──────────────────────────────────────────────────────────────────────────

def policy_logits(policy, obs):
    feats = policy.features_extractor(obs)
    latent_pi = policy.mlp_extractor.forward_actor(feats)
    return policy.action_net(latent_pi).view(obs.shape[0], ca.MOVES_PER_STEP, ca.N_MOVES)


def pretrain(model, O, G, epochs, tau, live, status, lr=3e-4, bs=1024, log=print):
    pol = model.policy
    dev = pol.device
    n = len(O)
    perm = np.random.default_rng(0).permutation(n)
    n_va = max(1000, n // 20)
    va, tr = perm[:n_va], perm[n_va:]
    Ot, Gt = torch.tensor(O, device=dev), torch.tensor(G, device=dev)
    params = list(pol.features_extractor.parameters()) + list(pol.mlp_extractor.policy_net.parameters()) + \
        list(pol.action_net.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    total = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)

    def loss_on(idx):
        lg = policy_logits(pol, Ot[idx])
        target = torch.softmax(Gt[idx] / tau, -1)
        logp = torch.log_softmax(lg, -1)
        return -(target.unsqueeze(1) * logp).sum(-1).mean(), lg

    k = 0
    t0 = time.time()
    for ep in range(epochs):
        pol.train()
        p = np.random.default_rng(ep).permutation(tr)
        for b in range(0, len(p), bs):
            idx = torch.tensor(p[b:b + bs], device=dev)
            loss, _ = loss_on(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            k += 1
        pol.eval()
        with torch.no_grad():
            idx = torch.tensor(va, device=dev)
            l_va, lg = loss_on(idx)
            g = Gt[idx]
            pick = lg[:, 0].argmax(-1)
            best = g.max(-1).values
            got = g.gather(1, pick.unsqueeze(1)).squeeze(1)
            top1 = (pick == g.argmax(-1)).float().mean().item()
            regret = (best - got).mean().item()
            improve = (got > 1e-6).float().mean().item()
            # 고른 이동의 크기 분포 (크게 바꾸는 판단을 배웠는지)
            mags = [abs(ca.move_label(int(a))[1]) for a in pick.tolist()]
        big = float(np.mean([m >= 0.15 for m in mags]))
        row = {"step": ep + 1, "t": round(time.time(), 1), "pre/loss": l_va.item(), "pre/top1": top1,
               "pre/regret": regret, "pre/improve": improve, "pre/big_move": big}
        write_live(live, row)
        status.update(phase_step=ep + 1, phase_total=epochs, phase_eta=(time.time() - t0) / (ep + 1) * (epochs - ep - 1))
        status.save()
        log(f"  [pretrain] epoch {ep + 1}/{epochs} · 검증 CE {l_va.item():.3f} · 최선 이동 일치 {top1:.0%} "
            f"· 고른 이동의 손해 {regret:.3f} · 개선되는 이동 {improve:.0%} · 큰 이동(|m|≥0.15) {big:.0%}", flush=True)
    return {"top1": top1, "regret": regret, "improve": improve, "big_move": big}


# ──────────────────────────────────────────────────────────────────────────
# 4. PPO + 평가
# ──────────────────────────────────────────────────────────────────────────

class CurveCallback(BaseCallback):
    def __init__(self, ctx):
        super().__init__()
        self.c = ctx
        self.episodes_done = deque(maxlen=2000)
        self.var_counts = np.zeros(len(VAR_KEYS))
        self.mag_counts = np.zeros(ca.N_MAGS)
        self.next_eval = ctx["eval_freq"]
        self.next_core = ctx["core_freq"]

    def _on_step(self):
        for a in np.asarray(self.locals["actions"]).ravel():
            self.var_counts[VAR_KEYS.index(_var_key(a))] += 1
            if a < ca.PIPE_UP:
                self.mag_counts[a % ca.N_MAGS] += 1
        for info in self.locals["infos"]:
            if "best_isp" in info:
                self.episodes_done.append(info)
        return True

    def _on_rollout_end(self):
        if self.episodes_done:
            eps = list(self.episodes_done)
            isps = [e["best_isp"] for e in eps if e["feasible"]]
            self.logger.record("ga/success_rate", np.mean([e["feasible"] for e in eps]))
            self.logger.record("ga/best_isp", np.mean(isps) if isps else 0.0)
            self.logger.record("ga/final_feasible", np.mean([e["final_feasible"] for e in eps]))
        if self.var_counts.sum():
            for k, v in zip(VAR_KEYS, self.var_counts / self.var_counts.sum()):
                self.logger.record(f"curve/var_{k}", float(v))
        if self.mag_counts.sum():
            for i, v in enumerate(self.mag_counts / self.mag_counts.sum()):
                self.logger.record(f"curve/mag_{i}", float(v))
            big = sum(self.mag_counts[i] for i, m in enumerate(ca.MAGS) if abs(m) >= 0.15)
            self.logger.record("curve/big_move", float(big / self.mag_counts.sum()))
        self.var_counts[:] = 0
        self.mag_counts[:] = 0
        if self.num_timesteps >= self.next_eval:
            self.next_eval = self.num_timesteps + self.c["eval_freq"]
            core = self.num_timesteps >= self.next_core
            if core:
                self.next_core = self.num_timesteps + self.c["core_freq"]
            evaluate(self.model, self.c, self.num_timesteps, core)


def evaluate(model, c, step, core):
    status = c["status"]
    predict = lambda o: model.predict(o, deterministic=True)[0]
    isps = agent_eval(c["eval_env"], predict, c["episodes"], c["eval_steps"])
    res = summarize(isps, c["base"])
    row = {"step": step, "elapsed_s": round(time.time() - c["t0"]), **{k: round(v, 4) for k, v in res.items()}}
    if res["score"] > c["best"]["sur"]:
        c["best"]["sur"] = res["score"]
        model.save(os.path.join(c["log_dir"], "best_sur_model"))
    msg = (f"[{step:>9,}] 대리모델 평가: 성공 {res['success']:.0%}  Isp {res['best_isp']:.2f}  "
           f"GA 대비 {res['isp_gain']:+.2f} s  승률 {res['win_rate']:.0%}")
    if core:
        prev = status.get("state")
        status.update(state="verify")
        status.save()
        rows = verify_core.verify(None, c["core_eps"], c["eval_steps"], env=c["core_env"], predict=predict,
                                  ga_rows=c["ga_rows"], pool=c["pool"])
        s = verify_core.summarize(rows, "agent")
        pr = verify_core.paired(rows)
        cs_ = verify_core.core_score(rows)
        row.update({"core_score": round(cs_, 4), "core_ok": round(s["core_ok"], 4),
                    "core_isp": round(s["core_isp_ok"], 3) if s["core_isp_ok"] else None,
                    "core_gain": round(pr["isp_gain"], 3) if pr["isp_gain"] is not None else None,
                    "core_win": round(pr["win_rate"], 4),
                    "pred_minus_core": round(s["pred_minus_core"], 3) if s["pred_minus_core"] is not None else None})
        with open(os.path.join(c["log_dir"], f"verify_{step}.json"), "w", encoding="utf-8") as f:
            json.dump({"step": step, "summary": s, "paired": pr, "rows": rows}, f, ensure_ascii=False, default=str)
        best = cs_ > c["best"]["core"]
        if best:
            c["best"]["core"] = cs_
            model.save(os.path.join(c["log_dir"], "best_model"))
            status["best"] = {"step": step, "core_score": cs_, "core_ok": s["core_ok"], "core_isp": s["core_isp_ok"],
                              "core_gain": pr["isp_gain"], "isp_gain": res["isp_gain"]}
        status.update(state=prev)
        msg += (f" | core: 합격 {s['core_ok']:.0%} (GA {c['ga_core']['core_ok']:.0%})  Isp {s['core_isp_ok'] or 0:.2f} "
                f"(GA {c['ga_core']['core_isp_ok'] or 0:.2f})  쌍 비교 {pr['isp_gain'] if pr['isp_gain'] is not None else float('nan'):+.2f} s  "
                f"예측−실제 {s['pred_minus_core'] or 0:+.2f} s{'  ★ best' if best else ''}")
    new = not os.path.exists(c["csv"])
    fields = ["step", "elapsed_s", "success", "best_isp", "isp_gain", "win_rate", "score", "core_score", "core_ok",
              "core_isp", "core_gain", "core_win", "pred_minus_core"]
    with open(c["csv"], "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})
    status.save()
    print(msg, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6_000_000)
    p.add_argument("--n-agents", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4, help="PPO 학습률 (사전학습 뒤라 작게)")
    p.add_argument("--lr-end-frac", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--len-min", type=int, default=20)
    p.add_argument("--len-max", type=int, default=60)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--eval-freq", type=int, default=500_000)
    p.add_argument("--core-freq", type=int, default=1_000_000)
    p.add_argument("--isp-penalty", type=float, default=0.5, help="점수 Isp = 예측 − k × 앙상블 표준편차")
    p.add_argument("--t-margin", type=float, default=1.0, help="예측 종료 시각이 simt − 이 값보다 늦으면 시간 초과")
    p.add_argument("--ok-threshold", type=float, default=0.9, help="대리모델의 core 정상 종료 확률이 이보다 낮으면 실패로 본다")
    p.add_argument("--teacher-states", type=int, default=150_000)
    p.add_argument("--teacher-eps", type=float, default=0.3)
    p.add_argument("--pretrain-epochs", type=int, default=30)
    p.add_argument("--pretrain-lr", type=float, default=5e-4)
    p.add_argument("--pretrain-bs", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.05, help="증류 온도 (점수 단위, 0.05 = Isp 1 s)")
    p.add_argument("--ts-model", default=None)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--init", default=None, help="사전학습 대신 이 모델에서 시작")
    args = p.parse_args(argv)

    from rl.ts_surrogate import MODEL_PATH
    ts_model = os.path.abspath(args.ts_model) if args.ts_model else MODEL_PATH
    name = args.name or "curve_" + time.strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(log_dir, exist_ok=True)
    cfg = dict(vars(args), kind="curve", ts_model=os.path.relpath(ts_model, HERE), n_envs=args.n_agents,
               eval_gens=args.eval_steps, gen_min=args.len_min, gen_max=args.len_max, obs_dim=ca.OBS_DIM,
               n_moves=ca.N_MOVES, mags=ca.MAGS, vars=VAR_KEYS,
               temperature=GA.load_fixed_values()["temperature"])
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    status = Status(os.path.join(log_dir, "status.json"), name=name, kind="curve", state="starting", pid=os.getpid(),
                    started=time.time(), total_steps=args.steps, step=0, config=cfg, vars=VAR_KEYS)
    status.save()
    live = LiveWriter(log_dir, status)

    env_kw = {"len_range": (args.len_min, args.len_max), "device": args.device, "model_path": ts_model,
              "isp_penalty": args.isp_penalty, "t_margin": args.t_margin, "ok_threshold": args.ok_threshold}
    env = ca.CurveVecEnv(args.n_agents, seed=args.seed, **env_kw)
    episodes = eval_episodes(env.impulse_range, env.thrust_range)
    eval_env = ca.CurveVecEnv(len(episodes) * EVAL_POP, seed=args.seed + 1, **env_kw)
    core_eps = verify_core.core_episodes()
    core_env = ca.CurveVecEnv(len(core_eps) * EVAL_POP, seed=args.seed + 2, **env_kw)
    pool = ProcessPoolExecutor(GA.N_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    print(f"실행 {name} | {args.device} | 에이전트 {args.n_agents} | 관측 {ca.OBS_DIM} · 이동 {ca.N_MOVES} × {ca.MOVES_PER_STEP}"
          f" | 대리모델 {cfg['ts_model']}\n  모니터링: http://127.0.0.1:10000/training#{name}", flush=True)

    try:
        # 1. 기준선
        status.update(state="baseline", phase="기본 GA 기준선 (대리모델 평가 + core 검증)")
        status.save()
        t = time.time()
        base = baseline_eval(env.simulator, episodes, 100)
        b = summarize(base, base)
        ga_rows = verify_core.ga_core_rows(env.simulator, core_eps, pool)
        ga_core = verify_core.summarize(ga_rows, "ga")
        status.update(baseline={k: b[k] for k in ("success", "best_isp", "score")}, eval_episodes=len(episodes),
                      ga_core={"core_ok": ga_core["core_ok"], "core_isp": ga_core["core_isp_ok"],
                               "core_score": verify_core.core_score(ga_rows, "ga"),
                               "pred_minus_core": ga_core["pred_minus_core"], "n": len(core_eps)})
        status.save()
        print(f"  기본 GA (대리모델, 32 × 100세대): 성공 {b['success']:.0%}  Isp {b['best_isp']:.2f} | core 검증: "
              f"합격 {ga_core['core_ok']:.0%}  Isp {ga_core['core_isp_ok'] or 0:.2f}  예측−실제 {ga_core['pred_minus_core'] or 0:+.2f} s"
              f"  ({time.time() - t:.0f}s)", flush=True)

        venv = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=args.gamma)
        lr0, lr1 = args.lr, args.lr * args.lr_end_frac
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*run PPO on the GPU.*")
            model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=6,
                        gamma=args.gamma, gae_lambda=0.95, learning_rate=lambda pr: lr1 + (lr0 - lr1) * pr,
                        ent_coef=args.ent_coef, target_kl=args.target_kl, seed=args.seed, device=args.device,
                        policy_kwargs=ca.policy_kwargs(args.d, args.layers), verbose=0)
        model.set_logger(Logger(log_dir, [TensorBoardOutputFormat(os.path.join(log_dir, "tb")), live]))

        if args.init:
            model.policy.load_state_dict(PPO.load(args.init, device=args.device).policy.state_dict())
            print(f"  초기 가중치: {args.init}", flush=True)
        else:
            # 2. 한 걸음 앞 보기
            status.update(state="teacher", phase="한 걸음 앞 보기 데이터 (이동 115가지 × 상태)")
            status.save()
            O, G = collect_teacher(env, args.teacher_states, args.teacher_eps, live, status)
            np.savez_compressed(os.path.join(log_dir, "teacher.npz"), O=O, G=G)
            # 3. 증류
            status.update(state="pretrain", phase="정책 사전학습 (앞 보기 → 정책 증류)")
            status.save()
            pre = pretrain(model, O, G, args.pretrain_epochs, args.tau, live, status, lr=args.pretrain_lr,
                           bs=args.pretrain_bs)
            status["pretrain"] = pre
            del O, G
            model.save(os.path.join(log_dir, "pretrained_model"))

        ctx = {"status": status, "eval_env": eval_env, "episodes": episodes, "eval_steps": args.eval_steps,
               "base": base, "log_dir": log_dir, "csv": os.path.join(log_dir, "progress.csv"), "t0": time.time(),
               "best": {"sur": -np.inf, "core": -np.inf}, "core_eps": core_eps, "core_env": core_env,
               "ga_rows": ga_rows, "ga_core": ga_core, "pool": pool,
               "eval_freq": args.eval_freq, "core_freq": args.core_freq}
        # 사전학습 직후 정책 (PPO 전) 평가
        status.update(state="running", phase="PPO 미세 조정")
        status.save()
        evaluate(model, ctx, 0, True)
        model.learn(total_timesteps=args.steps, callback=CurveCallback(ctx), progress_bar=False)
        evaluate(model, ctx, int(model.num_timesteps), True)
        status["state"] = "done"
    except KeyboardInterrupt:
        print("\n중단됨 — 지금까지의 정책을 저장합니다")
        status["state"] = "stopped"
    except Exception as exc:
        status.update(state="error", error=f"{type(exc).__name__}: {exc}")
        status.save()
        raise
    finally:
        if "model" in locals():
            model.save(os.path.join(log_dir, "final_model"))
            status.update(step=int(model.num_timesteps))
        status.update(finished=time.time())
        status.save()
        live.close()
        pool.shutdown(wait=False, cancel_futures=True)
    print(f"저장: {log_dir}/best_model.zip (core 기준), best_sur_model.zip, final_model.zip")


if __name__ == "__main__":
    main()
