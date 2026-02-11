import csv
import pathlib
import pickle

import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state

import eval_flow
import model as _model
import train_expert


def main():
    run_path = pathlib.Path("logs-bc/dummy-0qgv560m")
    level_paths = ["worlds/l/grasp_easy.json"]

    config = eval_flow.EvalConfig(
        num_evals=256,
        num_flow_steps=5,
        inference_delay=0,
        execute_horizon=1,
        method=eval_flow.NaiveMethodConfig(),
        model=_model.ModelConfig(
            action_chunk_size=4,
            channel_dim=64,
            channel_hidden_dim=128,
            token_hidden_dim=32,
            num_layers=2,
        ),
    )

    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP
    )
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    log_dirs = sorted([p for p in run_path.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    level_name = level_paths[0].replace("/", "_").replace(".json", "")
    with (log_dirs[-1] / "policies" / f"{level_name}.pkl").open("rb") as f:
        state_dict = pickle.load(f)

    obs_dim = jax.eval_shape(
        env.reset_to_level,
        jax.random.key(0),
        jax.tree.map(lambda x: x[0], levels),
        env_params,
    )[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    rng = jax.random.key(0)

    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=config.model, rngs=nnx.Rngs(rng))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    policy = nnx.merge(graphdef, state)

    info, video = eval_flow.eval(
        config, env, rng, jax.tree.map(lambda x: x[0], levels), policy, env_params, static_env_params
    )
    info = jax.device_get(info)
    video = jax.device_get(video)

    out_dir = pathlib.Path("exp_runs/stepA_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "method", "delay", "execute_horizon", *info.keys()])
        writer.writeheader()
        writer.writerow(
            {
                "level": level_paths[0],
                "method": "naive",
                "delay": 0,
                "execute_horizon": 1,
                **{k: float(v) for k, v in info.items()},
            }
        )

    imageio.mimwrite(out_dir / "preview.mp4", video, fps=15)
    print(out_dir)
    print(info)


if __name__ == "__main__":
    main()
