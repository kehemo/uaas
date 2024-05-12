import torch
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class Config:
    """
    Stores algorithmic hyperparameters.
    """

    discount: float = 0.995
    lr: float = 1e-3
    max_grad_norm: float = 0.5
    log_interval: int = 10
    max_episodes: int = 2000
    gae_lambda: float = 0.95
    use_critic: bool = False
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    train_ac_iters: int = 5
    use_discounted_reward: bool = False
    entropy_coef: float = 0.01
    use_gae: bool = False
    smooth_reward_window: int = 50


def compute_advantage_gae(values, rewards, T, gae_lambda, discount):
    """
    Compute Adavantage wiht GAE. See Section 4.4.2 in the lecture notes.

    values: value at each timestep (T,)
    rewards: reward obtained at each timestep.  Shape: (T,)
    T: the number of frames, float
    gae_lambda: hyperparameter, float
    discount: discount factor, float

    -----

    returns:

    advantages : tensor.float. Shape [T,]

                 gae advantage term for timesteps 0 to T

    """

    advantages = torch.zeros_like(values)
    #### TODO: populate GAE in advantages over T timesteps (10 pts) ############
    td = (rewards + discount * torch.roll(values, -1) - values)[:T]
    coeffs = torch.triu(
        (discount * gae_lambda)
        ** torch.cumsum(torch.triu(torch.ones((T, T)), diagonal=1), axis=1)
    ).to(td.device)
    advantages = coeffs @ td
    ############################################################################
    return advantages[:T]


def compute_discounted_return(rewards, discount, device=None):
    """
                rewards: reward obtained at timestep.  Shape: (T,)
                discount: discount factor. float

    ----
    returns: sum of discounted rewards. Shape: (T,)
    """
    returns = torch.zeros(*rewards.shape, device=device)
    #### TODO: populate discounted reward trajectory (10 pts) ############
    T = rewards.shape[0]
    coeffs = torch.triu(
        discount ** torch.cumsum(torch.triu(torch.ones((T, T)), diagonal=1), axis=1)
    ).to(rewards.device)
    returns = coeffs @ rewards
    ######################################################################

    return returns


def collect_experiences(env, acmodel, preprocess_obss, args, device=None):
    """
    Collects rollouts and computes advantages.

    -------
    env     : DoorKeyEnv

              The environement used to execute policies in.


    acmodel : ACModel

              The model used to evaluate observations to collect experiences

    args    : Config

              config arguments


    device  : torch.cuda.device

              the device torch tensors are evaluated on.

    -------

    Returns
    -------
    exps : dict
        Contains actions, rewards, advantages etc as attributes.
        Each attribute, e.g. `exps['reward']` has a shape
        (self.num_frames, ...).
    logs : dict
        Useful stats about the training process, including the average
        reward, policy loss, value loss, etc.
    """

    MAX_FRAMES_PER_EP = 300
    shape = (MAX_FRAMES_PER_EP,)

    actions = torch.zeros(*shape, device=device, dtype=torch.int)
    values = torch.zeros(*shape, device=device)
    rewards = torch.zeros(*shape, device=device)
    log_probs = torch.zeros(*shape, device=device)
    obss = [None] * MAX_FRAMES_PER_EP

    obs, _ = env.reset()

    total_return = 0

    T = 0

    while True:
        # Do one agent-environment interaction

        preprocessed_obs = preprocess_obss(obs, device=device)

        with torch.no_grad():
            dist, value = acmodel(preprocessed_obs)
        # action_probs = torch.softmax(dist.logits, dim=-1)
        # action = torch.tensor(0 if (action_probs[0] > action_probs[1]) else 1)
        action = dist.sample()[0]

        obss[T] = obs
        # update environment from taken action. We use the resulting observation,
        # reward, and whether or not environment is in the done/goal state.
        obs, reward, done, _, _ = env.step(action.item())

        # Update experiences values
        actions[T] = action
        values[T] = value
        rewards[T] = reward
        log_probs[T] = dist.log_prob(action)

        total_return += reward
        T += 1

        if done or T >= MAX_FRAMES_PER_EP - 1:
            break

    discounted_reward = compute_discounted_return(rewards[:T], args.discount, device)

    # dict containing information on the experience
    exps = dict(
        obs=preprocess_obss([obss[i] for i in range(T)], device=device),
        action=actions[:T],
        value=values[:T],
        reward=rewards[:T],
        advantage=discounted_reward - values[:T],
        log_prob=log_probs[:T],
        discounted_reward=discounted_reward,
        advantage_gae=compute_advantage_gae(
            values, rewards, T, args.gae_lambda, args.discount
        ),
    )

    logs = {"return_per_episode": total_return, "num_frames": T}

    return exps, logs


def run_experiment(env, acmodel, preprocess_obss, log_names, args, parameter_update):
    """
    Upper level function for running experiments to analyze reinforce and
    policy gradient methods. Instantiates a model, collects epxeriences, and
    then updates the neccessary parameters.

    args: Config arguments. dict
    paramter_update: function used to update model parameters
    seed: random seed. int

    return: DataFrame indexed by episode
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acmodel.to(device)

    # Smooth reward taken from last SMOOTH_REWARD_WINDOW timesteps
    SMOOTH_REWARD_WINDOW = args.smooth_reward_window

    pd_logs, rewards = [], [0] * SMOOTH_REWARD_WINDOW

    optimizer = torch.optim.Adam(acmodel.parameters(), lr=args.lr)
    num_frames = 0

    for update in tqdm(range(args.max_episodes)):
        # First collect experiences
        exps, logs1 = collect_experiences(env, acmodel, preprocess_obss, args, device)
        # update parameters from experiences
        logs2 = parameter_update(optimizer, acmodel, exps, args)

        logs = {**logs1, **logs2}

        num_frames += logs["num_frames"]

        rewards[update % SMOOTH_REWARD_WINDOW] = logs["return_per_episode"]
        smooth_reward = np.mean(rewards) if update >= SMOOTH_REWARD_WINDOW else np.nan

        data = {
            "episode": update,
            "num_frames": num_frames,
            "smooth_reward": smooth_reward,
            **{log_name: logs[log_name] for log_name in log_names},
        }

        pd_logs.append(data)

    return pd.DataFrame(pd_logs).set_index("episode")
