from typing import Type, Callable
import torch
import torch.nn as nn
from idil.baselines.IQLearn.utils.utils import (average_dicts, soft_update,
                                                hard_update, get_concat_samples)
from idil.baselines.IQLearn.iq import iq_loss, OFFLINE_METHOD_LOSS
from .option_softq import OptionSoftQ
from .option_sac import OptionSAC
from omegaconf import DictConfig


class IQMixin:

  def get_iq_variables(self, batch):
    'return vec_v_args, vec_next_v_args, vec_actions, done'
    raise NotImplementedError

  def iq_update_critic(self,
                       policy_batch,
                       expert_batch,
                       logger,
                       update_count,
                       use_target=False,
                       method_loss="value",
                       method_regularize=True,
                       method_div=""):
    if policy_batch is None:  # offline
      vec_v_args, vec_next_v_args, vec_actions, done = self.get_iq_variables(
          expert_batch)
      is_expert = torch.ones_like(expert_batch[-2], dtype=torch.bool)

      # for offline setting these shouldn't be changed
      method_loss = OFFLINE_METHOD_LOSS
      if method_regularize or method_div == "chi":
        # apply only one (same effect)
        method_regularize = False
        method_div = "chi"
    else:
      batch = get_concat_samples(policy_batch, expert_batch, False)
      vec_v_args, vec_next_v_args, vec_actions, done = self.get_iq_variables(
          batch[:-1])
      is_expert = batch[-1]

    agent = self

    current_Q = self.critic(*vec_v_args, *vec_actions, both=True)
    if isinstance(current_Q, tuple):
      q1_loss, loss_dict1 = iq_loss(agent, current_Q[0], vec_v_args,
                                    vec_next_v_args, vec_actions, done,
                                    is_expert, use_target, method_loss,
                                    method_regularize, method_div)
      q2_loss, loss_dict2 = iq_loss(agent, current_Q[1], vec_v_args,
                                    vec_next_v_args, vec_actions, done,
                                    is_expert, use_target, method_loss,
                                    method_regularize, method_div)
      critic_loss = 1 / 2 * (q1_loss + q2_loss)
      # merge loss dicts
      loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
      critic_loss, loss_dict = iq_loss(agent, current_Q, vec_v_args,
                                       vec_next_v_args, vec_actions, done,
                                       is_expert, use_target, method_loss,
                                       method_regularize, method_div)

    # logger.log('train/critic_loss', critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    if hasattr(self, 'clip_grad_val') and self.clip_grad_val:
      nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.clip_grad_val)
    # step critic
    self.critic_optimizer.step()
    return loss_dict

  def iq_update(self,
                policy_batch,
                expert_batch,
                logger,
                update_count,
                use_target=False,
                do_soft_update=False,
                method_loss="value",
                method_regularize=True,
                method_div=""):

    for _ in range(self.num_critic_update):
      losses = self.iq_update_critic(policy_batch, expert_batch, logger,
                                     update_count, use_target, method_loss,
                                     method_regularize, method_div)

    # args
    vdice_actor = False

    if self.actor:
      if not vdice_actor:

        vec_v_args_expert, _, _, _ = self.get_iq_variables(expert_batch)

        if policy_batch is None:
          vec_v_args = vec_v_args_expert
        else:
          # Use both policy and expert observations
          vec_v_args_policy, _, _, _ = self.get_iq_variables(policy_batch)
          vec_v_args = []
          for idx in range(len(vec_v_args_expert)):
            item = torch.cat([vec_v_args_policy[idx], vec_v_args_expert[idx]],
                             dim=0)
            vec_v_args.append(item)

        for i in range(self.num_actor_update):
          actor_alpha_losses = self.update_actor_and_alpha(
              *vec_v_args, logger, update_count)

        losses.update(actor_alpha_losses)

    if use_target and update_count % self.critic_target_update_frequency == 0:
      if do_soft_update:
        soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
      else:
        hard_update(self.critic_net, self.critic_target_net)

    return losses

  def iq_offline_update(self,
                        expert_batch,
                        logger,
                        update_count,
                        use_target=False,
                        do_soft_update=False,
                        method_regularize=True,
                        method_div=""):
    return self.iq_update(None, expert_batch, logger, update_count, use_target,
                          do_soft_update, OFFLINE_METHOD_LOSS,
                          method_regularize, method_div)


class IQLOptionSoftQ(IQMixin, OptionSoftQ):

  def __init__(self, config: DictConfig, num_inputs, action_dim, option_dim,
               discrete_obs, q_net_base: Type[nn.Module],
               cb_get_iq_variables: Callable):
    super().__init__(config, num_inputs, action_dim, option_dim, discrete_obs,
                     q_net_base)
    self.cb_get_iq_variables = cb_get_iq_variables
    self.method_loss = config.method_loss
    self.method_regularize = config.method_regularize
    self.method_div = config.method_div

  def get_iq_variables(self, batch):
    'return vec_v_args, vec_next_v_args, vec_actions, done'
    return self.cb_get_iq_variables(batch)


class IQLOptionSAC(IQMixin, OptionSAC):

  def __init__(self, config: DictConfig, obs_dim, action_dim, option_dim,
               discrete_obs, critic_base: Type[nn.Module], actor,
               cb_get_iq_variables: Callable):
    super().__init__(config, obs_dim, action_dim, option_dim, discrete_obs,
                     critic_base, actor)
    self.cb_get_iq_variables = cb_get_iq_variables
    self.method_loss = config.method_loss
    self.method_regularize = config.method_regularize
    self.method_div = config.method_div

  def get_iq_variables(self, batch):
    'return vec_v_args, vec_next_v_args, vec_actions, done'
    return self.cb_get_iq_variables(batch)
