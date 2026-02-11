import csv
import pathlib
import pickle
from dataclasses import dataclass

import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state

import eval_flow
import model as _model
import train_expert


@dataclass(frozen=True)
class RunConfig:
    run_path: str = "logs-bc/dummy-0qgv560m"
    output_root: str = "exp_runs/tail_v1"
    level_path: str = "worlds/l/grasp_easy.json"
    num_evals: int = 256
    num_flow_steps: int = 5
    save_preview: bool = False
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    methods: tuple[str, ...] = ("naive", "realtime")


def make_profiles() -> dict[str, eval_flow.DelayProfileConfig]:
    return {
        "long_tail": eval_flow.DelayProfileConfig(
            mode="mixture",
            p_spike=0.05,
            d_spike=4,
            base_delays=(0, 1),
            base_probs=(0.9, 0.1),
            tail_cap=None,
        ),
        "tail_controlled": eval_flow.DelayProfileConfig(
            mode="mixture",
            p_spike=0.05,
            d_spike=4,
            base_delays=(0, 1),
            base_probs=(0.9, 0.1),
            tail_cap=2,
        ),
    }


def load_policy(run_path: pathlib.Path, level_name: str, model_cfg: _model.ModelConfig, obs_dim: int, action_dim: int):
    log_dirs = sorted([p for p in run_path.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    with (log_dirs[-1] / "policies" / f"{level_name}.pkl").open("rb") as f:
        state_dict = pickle.load(f)

    policy = _model.FlowPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=model_cfg,
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def write_result(path: pathlib.Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    run_cfg = RunConfig()
    profiles = make_profiles()

    model_cfg = _model.ModelConfig(
        action_chunk_size=4,
        channel_dim=64,
        channel_hidden_dim=128,
        token_hidden_dim=32,
        num_layers=2,
    )

    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels([run_cfg.level_path], static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    level = jax.tree.map(lambda x: x[0], levels)

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level, env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    level_name = run_cfg.level_path.replace("/", "_").replace(".json", "")
    policy = load_policy(pathlib.Path(run_cfg.run_path), level_name, model_cfg, obs_dim, action_dim)

    summary_rows = []
    total = len(profiles) * len(run_cfg.methods) * len(run_cfg.seeds)
    done = 0
    for profile_name, profile_cfg in profiles.items():
        for method_name in run_cfg.methods:
            if method_name == "naive":
                method_cfg = eval_flow.NaiveMethodConfig()
            elif method_name == "realtime":
                method_cfg = eval_flow.RealtimeMethodConfig(prefix_attention_schedule="exp")
            else:
                raise ValueError(f"Unknown method: {method_name}")

            for seed in run_cfg.seeds:
                done += 1
                print(f"[{done}/{total}] profile={profile_name} method={method_name} seed={seed}")
                cfg = eval_flow.EvalConfig(
                    num_evals=run_cfg.num_evals,
                    num_flow_steps=run_cfg.num_flow_steps,
                    inference_delay=0,
                    delay_profile=profile_cfg,
                    execute_horizon=1,
                    method=method_cfg,
                    model=model_cfg,
                )
                info, video = eval_flow.eval(
                    cfg,
                    env,
                    jax.random.key(seed),
                    level,
                    policy,
                    env_params,
                    static_env_params,
                )
                info = {k: float(v) for k, v in jax.device_get(info).items()}
                out_row = {
                    "profile": profile_name,
                    "method": method_name,
                    "seed": seed,
                    "level": run_cfg.level_path,
                    "delay": "profile",
                    "execute_horizon": cfg.execute_horizon,
                    **info,
                }

                out_dir = pathlib.Path(run_cfg.output_root) / profile_name / method_name / f"seed_{seed}"
                write_result(out_dir / "results.csv", out_row)
                if run_cfg.save_preview:
                    imageio.mimwrite(out_dir / "preview.mp4", jax.device_get(video), fps=15)
                summary_rows.append(out_row)

    summary_path = pathlib.Path(run_cfg.output_root) / "results_all.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
