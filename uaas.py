import torch


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
    policy_loss = -torch.sum(logps * returns)
    ############################################

    return policy_loss


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


class UAASParameterUpdate:
    def __init__(self, alpha, epsilon):
        self.q_j = 0
        self.j = 1
        self.alpha = alpha
        self.epsilon = epsilon

    def step_size(self):
        return self.j ** (-0.5 + self.epsilon)

    def __call__(self, optimizer, acmodel, sb, args):
        """
        optimizer: Optimizer function used to perform gradient updates to model. torch.optim.Optimizer
        acmodel: Network used to compute policy. torch.nn.Module
        sb: stores experience data. Refer to "collect_experiences". dict
        args: Config arguments. Config

        return output logs : dict
        """
        dist, vals = acmodel(sb["obs"])
        logps = dist.log_prob(sb["action"])
        val_nograd = sb["value"]
        reward = sb["discounted_reward"]
        val_t1 = torch.roll(val_nograd, shifts=-1, dims=0)
        val_t1[-1] = 0
        reduced_reward = sb["reward"] + args.discount * val_t1
        score = (val_nograd - reward) * (val_nograd - reward)
        indices = []
        for x in score[1:]:
            s = x.item()
            self.q_j += self.step_size() * ((1 if self.q_j <= s else 0) - self.alpha)
            indices.append(1 if self.q_j <= s else 0)
        indices.append(0)
        reward_prime = torch.stack([reward, reduced_reward])[indices]
        advantage = reward_prime - val_nograd
        # computes policy loss
        policy_loss = compute_policy_loss_with_baseline(logps, advantage)
        update_policy_loss = policy_loss.item()

        value_loss = torch.norm(reward - vals, p=2)
        update_value_loss = value_loss.item()

        loss = value_loss + policy_loss

        # Update actor-critic
        optimizer.zero_grad()
        loss.backward()

        # Perform gradient clipping for stability
        update_grad_norm = (
            sum(p.grad.data.norm(2) ** 2 for p in acmodel.parameters()) ** 0.5
        )
        torch.nn.utils.clip_grad_norm_(acmodel.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log some values
        logs = {
            "policy_loss": update_policy_loss,
            "grad_norm": update_grad_norm,
            "value_loss": update_value_loss,
        }

        return logs


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
    reward = (reward - reward.mean()) / (reward.std() + 1e-10)
    ##############################################################################################

    # computes policy loss
    policy_loss = compute_policy_loss_reinforce(logps, reward)

    # Update actor-critic
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # Log some values
    logs = {"policy_loss": policy_loss.item()}

    return logs
