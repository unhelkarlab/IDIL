import torch
import torch.nn as nn
from idil_algs.baselines.IQLearn.utils.utils import (average_dicts, soft_update,
                                                     hard_update,
                                                     get_concat_samples)
from idil_algs.baselines.IQLearn.iq import iq_loss, OFFLINE_METHOD_LOSS
from .option_sac import OptionSAC
import time


class OptionIQL(OptionSAC):

  def iq_update_critic(self,
                       policy_batch,
                       expert_batch,
                       logger,
                       step,
                       use_target=False,
                       method_loss="value",
                       method_regularize=True,
                       method_div=""):

    if policy_batch is None:
      obs, prev_lat, prev_act, next_obs, latent, action, _, done = expert_batch
      is_expert = torch.ones_like(expert_batch[-2], dtype=torch.bool)

      # for offline setting these shouldn't be changed
      method_loss = OFFLINE_METHOD_LOSS
      method_regularize = False
    else:
      (obs, prev_lat, prev_act, next_obs, latent, action, _, done,
       is_expert) = get_concat_samples(policy_batch, expert_batch, False)

    vec_v_args = (obs, prev_lat, prev_act)
    vec_next_v_args = (next_obs, latent, action)
    vec_actions = (latent, action)

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
      nn.utils.clip_grad_norm_(self._critic.parameters(), self.clip_grad_val)
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

        if policy_batch is None:
          obs = expert_batch[0]
          prev_lat = expert_batch[1]
          prev_act = expert_batch[2]
        else:
          # Use both policy and expert observations
          obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)
          prev_lat = torch.cat([policy_batch[1], expert_batch[1]], dim=0)
          prev_act = torch.cat([policy_batch[2], expert_batch[2]], dim=0)

        for i in range(self.num_actor_update):
          actor_alpha_losses = self.update_actor_and_alpha(
              obs, prev_lat, prev_act, logger, update_count)

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
