import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm.notebook import tqdm
from gym_minigrid.envs.doorkey import DoorKeyEnv
import pandas as pd
import random
from tqdm import tqdm
from dataclasses import dataclass


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    """
    Initialize parameters of the network.
    m: torch.nn.Module
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def preprocess_obss(obss, device=None):
    """
    Convert observation into Torch.Tensor

    Parameters
    ----
    obss: dictionary or np.ndarray
    device: target device of torch.Tensor ('cpu', 'cuda')

    Return
    ----
    Torch Tensor
    """
    if isinstance(obss, dict):
        images = np.array([obss["image"]])
    else:
        images = np.array([o["image"] for o in obss])

    return torch.tensor(images, device=device, dtype=torch.float)


class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        self.render_mode = "rgb_array"
        super().__init__(size=5)

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1


@dataclass
class Config:
    """
    Stores algorithmic hyperparameters.
    """

    score_threshold = 0.93
    discount = 0.995
    lr = 1e-3
    max_grad_norm = 0.5
    log_interval = 10
    max_episodes = 2000
    gae_lambda = 0.95
    use_critic = False
    clip_ratio = 0.2
    target_kl = 0.01
    train_ac_iters = 5
    use_discounted_reward = False
    entropy_coef = 0.01
    use_gae = False


class ACModel(nn.Module):
    def __init__(self, num_actions, use_critic=False):  # what is num_actions
        """
        Represents an Actor Crictic model that takes a 2d, multi-channeled
        image as input.

        Parameters
        ----
        num_actions : int

                      The action space of the environment.
                      The action space for DoorKey5x5 is 7-dimensional:
                      0: turn left,
                      1: turn right,
                      2: forward,
                      3: pickup an object,
                      4: drop an object,
                      5: activate an object,
                      6: done completing task

        use_critics : bool

                      Critic network will be used in forward pass if flag is set
                      to true.
        """
        super().__init__()
        self.use_critic = use_critic

        # Define actor's model
        # Define CNN to process gym image
        self.image_conv_actor = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        # Take in output of actor CNN to output categorical distribution over
        # actions
        self.actor = nn.Sequential(
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, num_actions)
        )

        # Define critic's model
        if self.use_critic:
            # Define CNN to process gym image
            self.image_conv_critic = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
            )
            # Take in output of critic CNN to output a value representing the
            # reward
            self.critic = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

        # Initialize parameters correctly (don't remove this!)
        self.apply(init_params)

    def forward(self, obs):
        """
        Performs a forward pass through the actor-critic network

        Parameters
        ----
        obs : int tensor. Shape [Batch size, ImWidth, ImHeight, Channels]

              input to the network.
        ----

        returns:

        dist : torch.distribution
            The distribution of actions from policy. A Categorical distribution
            for discreet action spaces.
        value : torch.Tensor (Batch size, 1)
            value output by critic network
        """
        conv_in = obs.transpose(1, 3).transpose(2, 3)  # reshape into expected order

        dist, value = None, None
        ##### produce Categorical action distribtuion and critic value output #####
        ##### if self.use_critic is false, return all zeros for value ###################
        x = self.image_conv_actor(conv_in)
        embedding = x.reshape(x.shape[0], -1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        if self.use_critic:
            y = self.image_conv_critic(conv_in)
            embedding = y.reshape(y.shape[0], -1)

            value = self.critic(embedding).squeeze(1)
        else:
            value = torch.zeros((x.shape[0], 1), device=x.device)
        ##################################################################################

        return dist, value


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


def collect_experiences(env, acmodel, args, device=None):
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


def compute_policy_loss_reinforce(logps, returns):
    """
    Function for computing the policy loss for the REINFORCE algorithm. See
    4.2 of lecture notes.

                logps: log probabilities for each time step. Shape: (T,)
                returns: total return for each time step. Shape: (T,)

    ----
    return : tensor.float Shape: [T,]

             policy loss for each timestep
    """
    policy_loss = torch.tensor(0)

    #### TODO: complete policy loss (10 pts) ###
    # HINT:  Recall, that we want to perform gradient ASCENT to maximize returns
    policy_loss = -torch.mean(logps * returns)
    ############################################

    return policy_loss


def update_parameters_reinforce(optimizer, acmodel, sb, args):
    """
    optimizer: Optimizer function used to perform gradient updates to model. torch.optim.Optimizer
    acmodel: Network used to compute policy. torch.nn.Module
    sb: stores experience data. Refer to "collect_experiences". dict
    args: Config arguments. Config

    return output logs : dict
    """

    # logps is the log probability for taking an action for each time step. Shape (T,)
    logps, reward = None, None

    ### TODO: compute logps and reward from acmodel, sb['obs'], sb['action'], and sb['reward'] ###
    ### If args.use_discounted_reward is True, use sb['discounted_reward'] instead. ##############
    ### (10 pts) #########################################
    dist, val = acmodel(sb["obs"])
    logps = dist.log_prob(sb["action"])

    reward = sb["discounted_reward"] if args.use_discounted_reward else sb["reward"]
    ##############################################################################################

    # computes policy loss
    policy_loss = compute_policy_loss_reinforce(logps, reward)
    update_policy_loss = policy_loss.item()

    # Update actor-critic
    optimizer.zero_grad()
    policy_loss.backward()

    # Perform gradient clipping for stability
    for p in acmodel.parameters():
        if p.grad is None:
            print(
                "Make sure you're not instantiating any critic variables when the critic is not used"
            )
    update_grad_norm = (
        sum(p.grad.data.norm(2) ** 2 for p in acmodel.parameters()) ** 0.5
    )
    torch.nn.utils.clip_grad_norm_(acmodel.parameters(), args.max_grad_norm)
    optimizer.step()

    # Log some values
    logs = {"policy_loss": update_policy_loss, "grad_norm": update_grad_norm}

    return logs


def run_experiment(args, parameter_update, seed=0):
    """
    Upper level function for running experiments to analyze reinforce and
    policy gradient methods. Instantiates a model, collects epxeriences, and
    then updates the neccessary parameters.

    args: Config arguments. dict
    paramter_update: function used to update model parameters
    seed: random seed. int

    return: DataFrame indexed by episode
    """

    # Initialize random seeds for reproducible results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DoorKeyEnv5x5()

    acmodel = ACModel(env.action_space.n, use_critic=args.use_critic)
    acmodel.to(device)

    is_solved = False

    # Smooth reward taken from last SMOOTH_REWARD_WINDOW timesteps
    SMOOTH_REWARD_WINDOW = 50

    pd_logs, rewards = [], [0] * SMOOTH_REWARD_WINDOW

    optimizer = torch.optim.Adam(acmodel.parameters(), lr=args.lr)
    num_frames = 0

    pbar = tqdm(range(args.max_episodes))  # show progess bar ()
    for update in pbar:
        # First collect experiences
        exps, logs1 = collect_experiences(env, acmodel, args, device)
        # update parameters from experiences
        logs2 = parameter_update(optimizer, acmodel, exps, args)

        logs = {**logs1, **logs2}

        num_frames += logs["num_frames"]

        rewards.append(logs["return_per_episode"])

        smooth_reward = np.mean(rewards[-SMOOTH_REWARD_WINDOW:])

        data = {
            "episode": update,
            "num_frames": num_frames,
            "smooth_reward": smooth_reward,
            "reward": logs["return_per_episode"],
            "policy_loss": logs["policy_loss"],
        }

        if args.use_critic:
            data["value_loss"] = logs["value_loss"]

        pd_logs.append(data)

        #         pbar.set_postfix(data)

        # Early terminate
        if smooth_reward >= args.score_threshold:
            is_solved = True
            break

    if is_solved:
        print("Solved!")

    return pd.DataFrame(pd_logs).set_index("episode")


def compute_policy_loss_with_baseline(logps, advantages):
    """
    Computes policy loss with added baseline term. Refer to 4.3 in Lecture Notes.
    logps:  computed log probabilities. shape (T,)
    advantages: computed advantages. shape: (T,)

    ---

    return policy loss computed with baseline term: tensor.float. Shape (,1)

           refer to 4.3- Baseline in lecture notes

    """
    policy_loss = 0

    ### TODO: implement the policy loss (5 pts) ##############
    policy_loss = compute_policy_loss_reinforce(logps, advantages)
    ##################################################

    return policy_loss


def update_parameters_with_baseline(optimizer, acmodel, sb, args):
    """
    Updates model parameters using value and policy functions

    optimizer: Optimizer function used to perform gradient updates to model. torch.optim.Optimizer
    acmodel: Network used to compute policy. torch.nn.Module
    sb: stores experience data. Refer to "collect_experiences". dict
    args: Config arguments
    """

    def _compute_value_loss(values, returns):
        """
        Computes the value loss of critic model. See 4.3 of Lecture Notes

        values: computed values from critic model shape: (T,)
        returns: discounted rewards. shape: (T,)


        ---
        computes loss of value function. See 4.3, eq. 11 in lecture notes : tensor.float. Shape (,1)
        """

        value_loss = 0

        ### TODO: implement the value loss (5 pts) ###############
        value_loss = torch.norm(returns - values, p=2)
        ##################################################

        return value_loss

    logps, advantage, values, reward = None, None, None, None

    dist, values = acmodel(sb["obs"])
    logps = dist.log_prob(sb["action"])
    advantage = sb["advantage_gae"] if args.use_gae else sb["advantage"]
    reward = sb["discounted_reward"]

    policy_loss = compute_policy_loss_with_baseline(logps, advantage)
    value_loss = _compute_value_loss(values, reward)
    loss = policy_loss + value_loss

    update_policy_loss = policy_loss.item()
    update_value_loss = value_loss.item()

    # Update actor-critic
    optimizer.zero_grad()
    loss.backward()
    update_grad_norm = (
        sum(p.grad.data.norm(2) ** 2 for p in acmodel.parameters()) ** 0.5
    )
    torch.nn.utils.clip_grad_norm_(acmodel.parameters(), args.max_grad_norm)
    optimizer.step()

    # Log some values

    logs = {
        "policy_loss": update_policy_loss,
        "value_loss": update_value_loss,
        "grad_norm": update_grad_norm,
    }

    return logs
