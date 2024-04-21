import torch
import numpy as np
from .foo import *
from torch.distributions import Categorical
from tqdm import tqdm


def compute_v_i(values, rewards):
    """Compute the squared difference from estimated values and actual rewards"""
    return (values - rewards) ** 2


def update_quantile_threshold(q_prev, v_i, alpha, eta):
    """Update the quantile threshold for filtering trajectories"""
    condition = (v_i >= q_prev).float()
    return q_prev + eta * (condition - alpha)


def softclip(R, target_R):
    """Implement soft clipping logic here based on your specific requirements"""
    return torch.clamp(R, min=target_R)


def run_uaas_experiment(args, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DoorKeyEnv5x5()
    acmodel = ACModel(env.action_space.n, use_critic=True)
    acmodel.to(device)
    optimizer = torch.optim.Adam(acmodel.parameters(), lr=args.lr)

    alpha = 0.1  # Adjust this parameter as needed for the quantile update
    eta = 0.01  # Learning rate for quantile threshold update

    q_i = torch.tensor(0.0, device=device)  # Initial quantile threshold

    for episode in tqdm(range(args.max_episodes)):
        # Step 1: Collect trajectories
        exps, logs = collect_experiences(env, acmodel, args, device)

        # Split data into two subsets D_n and D_m
        num_data = exps["obs"].size(0)
        indices = torch.randperm(num_data).to(device)

        # Assuming you want to split 50-50 or adjust the ratio as needed
        split_point = round(num_data * 0.5)
        D_n_indices = indices[:split_point]
        D_m_indices = indices[split_point:]

        D_n = {key: val[D_n_indices] for key, val in exps.items()}
        D_m = {key: val[D_m_indices] for key, val in exps.items()}

        # Compute v_i and update q_i using only D_m
        v_i = compute_v_i(D_m["value"], D_m["reward"])
        q_i = update_quantile_threshold(q_i, v_i, alpha, eta)

        # Update V_phi using D_m
        value_loss = torch.mean(v_i)
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

        # Compute adjusted rewards R'^gamma
        R_gamma = D_n["reward"]  # Assuming you've computed discounted rewards
        adjusted_rewards = torch.where(
            v_i <= q_i, R_gamma, softclip(R_gamma, D_n["reward"], episode)
        )

        # Compute policy gradient and update policy
        dists = Categorical(logits=F.log_softmax(acmodel(D_n["obs"]).logits, dim=1))
        log_probs = dists.log_prob(D_n["action"])
        policy_loss = -torch.mean(log_probs * adjusted_rewards)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % args.log_interval == 0:
            print(f"Episode {episode}: Loss = {policy_loss.item()}")

    print("Experiment completed.")


# Example usage
# args = Config(max_episodes=1000, lr=0.01, log_interval=100)
# run_uaas_experiment(args)
