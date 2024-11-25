import idil.baselines.IQLearn.agent.sac_models as iqlm
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.autograd import Variable, grad
from idil.baselines.IQLearn.utils.utils import mlp, weight_init
from omegaconf import DictConfig


class OptionSACQCritic(nn.Module):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__()
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim
    self.use_tanh = False
    self.gamma = config.gamma
    self.use_prev_action = config.use_prev_action

  def _get_input(self, obs, prev_lat, prev_act, lat, act):
    if self.use_prev_action:
      input = [obs, prev_lat, prev_act, lat, act]
    else:
      input = [obs, prev_lat, lat, act]

    return input


class OptionDoubleQCritic(OptionSACQCritic):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__(config, obs_dim, action_dim, lat_dim)

    list_hidden_dims = config.hidden_critic
    input_dim = (obs_dim + lat_dim + lat_dim + action_dim +
                 int(config.extra_option_dim))

    if self.use_prev_action:
      input_dim += action_dim + int(config.extra_action_dim)

    # Q1 architecture
    self.Q1 = mlp(input_dim, 1, list_hidden_dims)

    # Q2 architecture
    self.Q2 = mlp(input_dim, 1, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs, prev_lat, prev_act, lat, act, both=False):

    q_input = torch.cat(self._get_input(obs, prev_lat, prev_act, lat, act),
                        dim=-1)
    q1 = self.Q1(q_input)
    q2 = self.Q2(q_input)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.min(q1, q2)

  def grad_pen(self,
               obs1,
               prev_lat1,
               prev_act1,
               lat1,
               act1,
               obs2,
               prev_lat2,
               prev_act2,
               lat2,
               act2,
               lambda_=1):
    expert_data = torch.cat(
        self._get_input(obs1, prev_lat1, prev_act1, lat1, act1), 1)
    policy_data = torch.cat(
        self._get_input(obs2, prev_lat2, prev_act2, lat2, act2), 1)

    alpha = torch.rand(expert_data.size()[0], 1)
    alpha = alpha.expand_as(expert_data).to(expert_data.device)

    interpolated = alpha * expert_data + (1 - alpha) * policy_data
    interpolated = Variable(interpolated, requires_grad=True)

    prev_act_dim = self.action_dim if self.use_prev_action else 0
    split_dim = [
        self.obs_dim, self.lat_dim, prev_act_dim, self.lat_dim, self.action_dim
    ]

    tup_int_input = torch.split(interpolated, split_dim, dim=1)
    q = self.forward(*tup_int_input, both=True)
    ones = torch.ones(q[0].size()).to(policy_data.device)
    gradient = grad(
        outputs=q,
        inputs=interpolated,
        grad_outputs=[ones, ones],
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen


class OptionSingleQCritic(OptionSACQCritic):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__(config, obs_dim, action_dim, lat_dim)

    list_hidden_dims = config.hidden_critic
    input_dim = (obs_dim + lat_dim + lat_dim + action_dim +
                 int(config.extra_option_dim))

    if self.use_prev_action:
      input_dim += action_dim + int(config.extra_action_dim)

    # Q1 architecture
    self.Q1 = mlp(input_dim, 1, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs, prev_lat, prev_act, lat, act, both=False):

    q_input = torch.cat(self._get_input(obs, prev_lat, prev_act, lat, act),
                        dim=-1)
    q1 = self.Q1(q_input)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)

    return q1

  def grad_pen(self,
               obs1,
               prev_lat1,
               prev_act1,
               lat1,
               act1,
               obs2,
               prev_lat2,
               prev_act2,
               lat2,
               act2,
               lambda_=1):
    expert_data = torch.cat(
        self._get_input(obs1, prev_lat1, prev_act1, lat1, act1), 1)
    policy_data = torch.cat(
        self._get_input(obs2, prev_lat2, prev_act2, lat2, act2), 1)

    alpha = torch.rand(expert_data.size()[0], 1)
    alpha = alpha.expand_as(expert_data).to(expert_data.device)

    interpolated = alpha * expert_data + (1 - alpha) * policy_data
    interpolated = Variable(interpolated, requires_grad=True)

    prev_act_dim = self.action_dim if self.use_prev_action else 0
    split_dim = [
        self.obs_dim, self.lat_dim, prev_act_dim, self.lat_dim, self.action_dim
    ]

    tup_int_input = torch.split(interpolated, split_dim, dim=1)
    q = self.forward(*tup_int_input, both=True)
    ones = torch.ones(q[0].size()).to(policy_data.device)
    gradient = grad(
        outputs=q,
        inputs=interpolated,
        grad_outputs=[ones, ones],
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen


class AbstractOptionActor(nn.Module):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__()

    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim
    list_hidden_dims = config.hidden_policy
    self.use_nn_logstd = config.use_nn_logstd

    input_dim = self.obs_dim + self.lat_dim
    output_dim = self._get_output_dim()

    self.trunk = mlp(input_dim, output_dim, list_hidden_dims)

    self.outputs = dict()
    self.apply(weight_init)

  def _get_output_dim(self):
    raise NotImplementedError

  def forward(self, obs, lat):
    raise NotImplementedError

  def rsample(self, obs, lat):
    raise NotImplementedError

  def sample(self, obs, lat):
    raise NotImplementedError

  def exploit(self, obs, lat):
    raise NotImplementedError

  def is_discrete(self):
    raise NotImplementedError

  def evaluate_action(self, obs, lat, action):
    raise NotImplementedError


class SoftDiscreteOptionActor(AbstractOptionActor):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__(config, obs_dim, action_dim, lat_dim)
    self.temperature = config.gumbel_temperature

  def _get_output_dim(self):
    return self.action_dim

  def forward(self, obs, lat):
    logits = self.trunk(torch.cat((obs, lat), dim=-1))
    dist = iqlm.GumbelSoftmax(self.temperature, logits=logits)
    return dist

  def rsample(self, obs, lat):
    dist = self.forward(obs, lat)
    action = dist.rsample()
    log_prob = dist.log_prob(action).view(-1, 1)

    return action, log_prob

  def sample(self, obs, lat):
    dist = self.forward(obs, lat)
    samples = dist.sample()
    log_probs = dist.log_prob(samples).view(-1, 1)

    return samples, log_probs

  def evaluate_action(self, obs, lat, action):
    dist = self.forward(obs, lat)
    if action.shape[-1] != self.action_dim:
      action = action.reshape(-1)
    log_probs = dist.log_prob(action).view(-1, 1)

    return log_probs

  def exploit(self, obs, lat):
    dist = self.forward(obs, lat)
    return dist.logits.argmax(dim=-1)

  def is_discrete(self):
    return True


class DiagGaussianOptionActor(AbstractOptionActor):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__(config, obs_dim, action_dim, lat_dim)
    self.log_std_bounds = config.log_std_bounds
    self.bounded = config.bounded_actor
    self.clamp_action_logstd = config.clamp_action_logstd
    if not self.use_nn_logstd:
      self.action_logstd = nn.Parameter(
          torch.empty(1, self.action_dim, dtype=torch.float32).fill_(0.))

  def _get_output_dim(self):
    if self.use_nn_logstd:
      return 2 * self.action_dim
    else:
      return self.action_dim

  def forward(self, obs, lat):
    if self.use_nn_logstd:
      mu, log_std = self.trunk(torch.cat((obs, lat), dim=-1)).chunk(2, dim=-1)
    else:
      mu = self.trunk(torch.cat((obs, lat), dim=-1))
      log_std = self.action_logstd.expand_as(mu)

    if self.clamp_action_logstd:
      log_std = log_std.clamp(self.log_std_bounds[0], self.log_std_bounds[1])
    else:
      # restrict the range of log_std for numerical stability
      log_std = torch.tanh(log_std)
      log_std_min, log_std_max = self.log_std_bounds
      log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    std = log_std.exp()

    if self.bounded:
      dist = iqlm.SquashedNormal(mu, std)
    else:
      mu = mu.clamp(-10, 10)
      dist = Normal(mu, std)

    return dist

  def rsample(self, obs, lat):
    dist = self.forward(obs, lat)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    return action, log_prob

  def sample(self, obs, lat):
    return self.rsample(obs, lat)

  def evaluate_action(self, obs, lat, action):
    dist = self.forward(obs, lat)
    if self.bounded:
      EPS = 1.e-7
      action = action.clip(-1.0 + EPS, 1.0 - EPS)
    log_probs = dist.log_prob(action).sum(-1, keepdim=True)

    return log_probs

  def exploit(self, obs, lat):
    return self.forward(obs, lat).mean

  def is_discrete(self):
    return False


class AbstractOptionThinker(nn.Module):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__()

    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim
    self.use_prev_action = config.use_prev_action

    input_dim = self.obs_dim + self.lat_dim + int(config.extra_option_dim)
    if self.use_prev_action:
      input_dim += self.action_dim + int(config.extra_action_dim)

    output_dim = self.lat_dim

    self.trunk = mlp(input_dim, output_dim, config.hidden_option)

    self.outputs = dict()
    self.apply(weight_init)

  def forward(self, obs, prev_lat, prev_act):
    raise NotImplementedError

  def rsample(self, obs, prev_lat, prev_act):
    raise NotImplementedError

  def sample(self, obs, prev_lat, prev_act):
    raise NotImplementedError

  def exploit(self, obs, prev_lat, prev_act):
    raise NotImplementedError

  def is_discrete(self):
    raise NotImplementedError


class SoftDiscreteOptionThinker(AbstractOptionThinker):

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim):
    super().__init__(config, obs_dim, action_dim, lat_dim)
    self.temperature = config.gumbel_temperature

  def forward(self, obs, prev_lat, prev_act):
    if self.use_prev_action:
      input = (obs, prev_lat, prev_act)
    else:
      input = (obs, prev_lat)

    logits = self.trunk(torch.cat(input, dim=-1))
    dist = iqlm.GumbelSoftmax(self.temperature, logits=logits)
    return dist

  def rsample(self, obs, prev_lat, prev_act):
    dist = self.forward(obs, prev_lat, prev_act)
    action = dist.rsample()
    log_prob = dist.log_prob(action).view(-1, 1)

    return action, log_prob

  def sample(self, obs, prev_lat, prev_act):
    dist = self.forward(obs, prev_lat, prev_act)
    samples = dist.sample()
    log_probs = dist.log_prob(samples).view(-1, 1)

    return samples, log_probs

  def exploit(self, obs, prev_lat, prev_act):
    dist = self.forward(obs, prev_lat, prev_act)
    return dist.logits.argmax(dim=-1)

  def mental_probs(self, obs, prev_lat, prev_act):
    dist = self.forward(obs, prev_lat, prev_act)
    probs = dist.probs
    # avoid numerical instability
    z = (probs == 0.0).float() * 1e-10
    log_probs = torch.log(probs + z)

    return probs, log_probs

  def is_discrete(self):
    return True
