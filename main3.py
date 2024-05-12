from foo import *
from cartpole import *
from uaas import *
import sys


for name, param_update, use_gae in [
    ("bl", update_parameters_with_baseline, False),
    *[
        (f"uaas{alpha}", UAASParameterUpdate(alpha, 0.1), False)
        for alpha in [0.01, 0.02, 0.05, 0.1, 0.2]
    ],
    ("gae", update_parameters_with_baseline, True),
]:
    smooth_reward_window = 300
    args = Config(
        use_critic=True,
        max_episodes=3000,
        lr=1e-3,
        smooth_reward_window=smooth_reward_window,
        use_gae=use_gae,
        max_frames_per_ep=500,
    )
    env = make_env()
    acmodel = Policy(env.action_space.n, use_critic=args.use_critic)
    df = run_experiment(
        env,
        acmodel,
        preprocess_obss,
        ["return_per_episode"],
        args,
        param_update,
    )
    df.to_csv(f"{sys.argv[1]}_{name}_{sys.argv[2]}.csv")

    reward_total = 0
    for _ in range(smooth_reward_window):
        exps, logs = collect_experiences(env, acmodel, preprocess_obss, args)
        reward_total += logs["return_per_episode"]
    with open(f"{sys.argv[1]}_{name}_{sys.argv[2]}_final.txt", "w+") as f:
        f.write(f"{reward_total / smooth_reward_window}\n")
