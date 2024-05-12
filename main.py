from foo import *
from blackjack import *
from uaas import *
import sys


for name, param_update, use_gae in [
    ("bl", update_parameters_with_baseline, False),
    ("uaas", UAASParameterUpdate(0.01, 0.1), False),
    ("gae", update_parameters_with_baseline, True),
]:
    smooth_reward_window = 5000
    args = Config(
        use_critic=True,
        max_episodes=25000,
        smooth_reward_window=smooth_reward_window,
        lr=5e-7,
        use_gae=use_gae,
    )
    env = make_balckjack_env()
    acmodel = BlackjackACModel(env.action_space.n, use_critic=args.use_critic)
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
