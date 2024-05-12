from foo import *
from cartpole import *
from uaas import *
import sys

smooth_reward_window = 50
args = Config(
    use_critic=False,
    max_episodes=1000,
    smooth_reward_window=smooth_reward_window,
    lr=1e-2,
    use_gae=False,
)
env = make_env()
acmodel = Policy(env.action_space.n, use_critic=args.use_critic)
df = run_experiment(
    env,
    acmodel,
    preprocess_obss,
    ["return_per_episode"],
    args,
    update_parameters_reinforce,
)
