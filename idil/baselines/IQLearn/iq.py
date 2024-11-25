"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch
import torch.nn.functional as F

OFFLINE_METHOD_LOSS = "value_expert"


# Full IQ-Learn objective with other divergences and options
def iq_loss(agent,
            current_Q,
            vec_v_args,
            vec_next_v_args,
            vec_actions,
            done,
            is_expert,
            use_target,
            method_loss="value",
            method_regularize=True,
            method_div=""):
  # args
  method_type = "iq"
  method_grad_pen = False
  method_lambda_gp = 10
  method_alpha = 0.5

  gamma = agent.gamma
  loss_dict = {}

  current_v = agent.getV(*vec_v_args)
  if use_target:
    with torch.no_grad():
      next_v = agent.get_targetV(*vec_next_v_args)
  else:
    next_v = agent.getV(*vec_next_v_args)

  # keep track of value of initial states
  vec_v_args_expert = []
  for arg in vec_v_args:
    vec_v_args_expert.append(arg[is_expert.squeeze(1), ...])

  v0 = agent.getV(*vec_v_args_expert).mean()
  loss_dict['v0'] = v0.item()

  #  calculate 1st term for IQ loss
  #  -E_(ρ_expert)[Q(s, a) - γV(s')]
  y = (1 - done) * gamma * next_v
  reward = (current_Q - y)[is_expert]

  if method_div == "hellinger":
    phi_reward = reward / (1 + reward)
  elif method_div == "kl":
    # original dual form for kl divergence
    phi_reward = -torch.exp(-reward - 1)
  elif method_div == "js":
    # jensen–shannon
    phi_reward = torch.log(2 - torch.exp(-reward))
  elif method_div == "chi":  # also for offline learning setting
    phi_reward = reward - 1 / (4 * method_alpha) * reward**2
  else:
    phi_reward = reward

  # phi_reward = torch.clamp(phi_reward, min=-1e20, max=1e20)
  loss = -phi_reward.mean()

  loss_dict['softq_loss'] = loss.item()

  # calculate 2nd term for IQ loss, we show different sampling strategies
  if method_loss == "value_expert":
    # sample using only expert states (works offline)
    # E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (current_v - y)[is_expert].mean()
    loss += value_loss
    loss_dict['value_loss'] = value_loss.item()

  elif method_loss == "value":
    # sample using expert and policy states (works online)
    # E_(ρ)[V(s) - γV(s')]
    value_loss = (current_v - y).mean()
    loss += value_loss
    loss_dict['value_loss'] = value_loss.item()

  elif method_loss == "v0":
    # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
    # (1-γ)E_(ρ0)[V(s0)]
    v0_loss = (1 - gamma) * v0
    loss += v0_loss
    loss_dict['v0_loss'] = v0_loss.item()

  else:
    raise ValueError(f'This sampling method is not implemented: {method_type}')

  if method_grad_pen:
    # add a gradient penalty to loss (Wasserstein_1 metric)
    vec_actions_expert = []
    vec_actions_policy = []
    for item in vec_actions:
      vec_actions_expert.append(item[is_expert.squeeze(1), ...])
      vec_actions_policy.append(item[~is_expert.squeeze(1), ...])

    vec_v_args_policy = []
    for arg in vec_v_args:
      vec_v_args_policy.append(arg[~is_expert.squeeze(1), ...])

    gp_loss = agent.critic_net.grad_pen(*vec_v_args_expert, *vec_actions_expert,
                                        *vec_v_args_policy, *vec_actions_policy,
                                        method_lambda_gp)
    loss_dict['gp_loss'] = gp_loss.item()
    loss += gp_loss

  if method_regularize:
    # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
    y = (1 - done) * gamma * next_v

    reward = current_Q - y
    chi2_loss = 1 / (4 * method_alpha) * (reward**2).mean()
    loss += chi2_loss
    loss_dict['regularize_loss'] = chi2_loss.item()

  loss_dict['total_loss'] = loss.item()
  return loss, loss_dict
