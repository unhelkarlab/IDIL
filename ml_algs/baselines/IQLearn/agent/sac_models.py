import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.autograd import Variable, grad
from torch.distributions import Categorical, RelaxedOneHotCategorical
from ..utils.utils import mlp, weight_init


class SACQCritic(nn.Module):

  def __init__(self,
               obs_dim,
               action_dim,
               list_hidden_dims,
               gamma=0.99,
               double_q: bool = False,
               use_tanh: bool = False):
    super().__init__()
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.use_tanh = use_tanh
    self.gamma = gamma
    self.double_q = double_q


class DoubleQCritic(SACQCritic):

  def __init__(self,
               obs_dim,
               action_dim,
               list_hidden_dims,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, list_hidden_dims, gamma, True,
                     use_tanh)

    # Q1 architecture
    self.Q1 = mlp(obs_dim + action_dim, 1, list_hidden_dims)

    # Q2 architecture
    self.Q2 = mlp(obs_dim + action_dim, 1, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs, action, both=False):
    assert obs.size(0) == action.size(0)

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.min(q1, q2)

  def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
    expert_data = torch.cat([obs1, action1], 1)
    policy_data = torch.cat([obs2, action2], 1)

    alpha = torch.rand(expert_data.size()[0], 1)
    alpha = alpha.expand_as(expert_data).to(expert_data.device)

    interpolated = alpha * expert_data + (1 - alpha) * policy_data
    interpolated = Variable(interpolated, requires_grad=True)

    interpolated_state, interpolated_action = torch.split(
        interpolated, [self.obs_dim, self.action_dim], dim=1)
    q = self.forward(interpolated_state, interpolated_action, both=True)
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


class DoubleQCriticMax(SACQCritic):

  def __init__(self,
               obs_dim,
               action_dim,
               list_hidden_dims,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, list_hidden_dims, gamma, True,
                     use_tanh)

    # Q1 architecture
    self.Q1 = mlp(obs_dim + action_dim, 1, list_hidden_dims)

    # Q2 architecture
    self.Q2 = mlp(obs_dim + action_dim, 1, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs, action, both=False):
    assert obs.size(0) == action.size(0)

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.max(q1, q2)


class SingleQCritic(SACQCritic):

  def __init__(self,
               obs_dim,
               action_dim,
               list_hidden_dims,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, list_hidden_dims, gamma, False,
                     use_tanh)

    # Q architecture
    self.Q = mlp(obs_dim + action_dim, 1, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs, action, both=False):
    assert obs.size(0) == action.size(0)

    obs_action = torch.cat([obs, action], dim=-1)
    q = self.Q(obs_action)

    if self.use_tanh:
      q = torch.tanh(q) * 1 / (1 - self.gamma)

    return q

  def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
    expert_data = torch.cat([obs1, action1], 1)
    policy_data = torch.cat([obs2, action2], 1)

    alpha = torch.rand(expert_data.size()[0], 1)
    alpha = alpha.expand_as(expert_data).to(expert_data.device)

    interpolated = alpha * expert_data + (1 - alpha) * policy_data
    interpolated = Variable(interpolated, requires_grad=True)

    interpolated_state, interpolated_action = torch.split(
        interpolated, [self.obs_dim, self.action_dim], dim=1)
    q = self.forward(interpolated_state, interpolated_action)
    ones = torch.ones(q.size()).to(policy_data.device)
    gradient = grad(
        outputs=q,
        inputs=interpolated,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen


class DoubleQCriticState(SACQCritic):

  def __init__(self,
               obs_dim,
               action_dim,
               list_hidden_dims,
               gamma=0.99,
               use_tanh: bool = False):
    super().__init__(obs_dim, action_dim, list_hidden_dims, gamma, True,
                     use_tanh)

    # Q1 architecture
    self.Q1 = mlp(obs_dim, 1, list_hidden_dims)

    # Q2 architecture
    self.Q2 = mlp(obs_dim, 1, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs, action, both=False):
    assert obs.size(0) == action.size(0)

    q1 = self.Q1(obs)
    q2 = self.Q2(obs)

    if self.use_tanh:
      q1 = torch.tanh(q1) * 1 / (1 - self.gamma)
      q2 = torch.tanh(q2) * 1 / (1 - self.gamma)

    if both:
      return q1, q2
    else:
      return torch.min(q1, q2)

  def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
    expert_data = obs1
    policy_data = obs2

    alpha = torch.rand(expert_data.size()[0], 1)
    alpha = alpha.expand_as(expert_data).to(expert_data.device)

    interpolated = alpha * expert_data + (1 - alpha) * policy_data
    interpolated = Variable(interpolated, requires_grad=True)

    interpolated_state, interpolated_action = torch.split(
        interpolated, [self.obs_dim, self.action_dim], dim=1)
    q = self.forward(interpolated_state, interpolated_action)
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


class TanhTransform(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
      mu = tr(mu)
    return mu


class GumbelSoftmax(RelaxedOneHotCategorical):
  '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

  def sample(self, sample_shape=torch.Size()):
    '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
    u = torch.empty(self.logits.size(),
                    device=self.logits.device,
                    dtype=self.logits.dtype).uniform_(0, 1)
    noisy_logits = self.logits - torch.log(-torch.log(u))
    return torch.argmax(noisy_logits, dim=-1)

  def rsample(self, sample_shape=torch.Size()):
    '''
      ref: https://github.com/kengz/SLM-Lab/blob/master/slm_lab/lib/distribution.py
      Gumbel-softmax resampling using the Straight-Through trick.
      Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
      '''
    rout = super().rsample(sample_shape)  # differentiable
    out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
    return (out - rout).detach() + rout

  def log_prob(self, value):
    '''value is one-hot or relaxed'''
    if value.shape != self.logits.shape:
      value = F.one_hot(value.long(), self.logits.shape[-1]).float()
      assert value.shape == self.logits.shape
    return -torch.sum(-value * F.log_softmax(self.logits, -1), -1)


class AbstractActor(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, obs):
    raise NotImplementedError

  def rsample(self, obs):
    raise NotImplementedError

  def sample(self, obs):
    raise NotImplementedError

  def exploit(self, obs):
    raise NotImplementedError

  def is_discrete(self):
    raise NotImplementedError


class DiagGaussianActor(AbstractActor):
  """torch.distributions implementation of an diagonal Gaussian policy."""

  def __init__(self,
               obs_dim,
               action_dim,
               list_hidden_dims,
               log_std_bounds,
               bounded=True,
               use_nn_logstd=False,
               clamp_action_logstd=False):
    super().__init__()
    self.use_nn_logstd = use_nn_logstd
    self.clamp_action_logstd = clamp_action_logstd

    output_dim = action_dim
    if self.use_nn_logstd:
      output_dim = 2 * action_dim
    else:
      self.action_logstd = nn.Parameter(
          torch.empty(1, action_dim, dtype=torch.float32).fill_(0.))

    self.trunk = mlp(obs_dim, output_dim, list_hidden_dims)

    self.apply(weight_init)

    self.log_std_bounds = log_std_bounds
    self.bounded = bounded

  def forward(self, obs):
    if self.use_nn_logstd:
      mu, log_std = self.trunk(obs).chunk(2, dim=-1)
    else:
      mu = self.trunk(obs)
      log_std = self.action_logstd.expand_as(mu)

    # clamp logstd
    if self.clamp_action_logstd:
      log_std = log_std.clamp(self.log_std_bounds[0], self.log_std_bounds[1])
    else:
      # constrain log_std inside [log_std_min, log_std_max]
      log_std = torch.tanh(log_std)
      log_std_min, log_std_max = self.log_std_bounds
      log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    std = log_std.exp()

    if self.bounded:
      dist = SquashedNormal(mu, std)
    else:
      mu = mu.clamp(-10, 10)
      dist = pyd.Normal(mu, std)

    return dist

  def rsample(self, obs):
    dist = self.forward(obs)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    return action, log_prob

  def sample(self, obs):
    return self.rsample(obs)

  def exploit(self, obs):
    return self.forward(obs).mean

  def is_discrete(self):
    return False


class DiscreteActor(AbstractActor):
  'cf) https://github.com/openai/spinningup/issues/148 '

  def __init__(self, obs_dim, action_dim, list_hidden_dims):
    super().__init__()

    output_dim = action_dim
    self.trunk = mlp(obs_dim, output_dim, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs):
    logits = self.trunk(obs)
    dist = Categorical(logits=logits)
    return dist

  def action_probs(self, obs):
    dist = self.forward(obs)
    action_probs = dist.probs
    # avoid numerical instability
    z = (action_probs == 0.0).float() * 1e-10
    log_action_probs = torch.log(action_probs + z)

    return action_probs, log_action_probs

  def exploit(self, obs):
    dist = self.forward(obs)
    return dist.logits.argmax(dim=-1)

  def sample(self, obs):
    dist = self.forward(obs)

    samples = dist.sample()
    action_log_probs = dist.log_prob(samples)

    return samples, action_log_probs

  def rsample(self, obs):
    'should not be used'
    raise NotImplementedError

  def is_discrete(self):
    return True


class SoftDiscreteActor(AbstractActor):
  'cf) https://github.com/openai/spinningup/issues/148 '

  def __init__(self, obs_dim, action_dim, list_hidden_dims, temperature):
    super().__init__()

    output_dim = action_dim
    self.trunk = mlp(obs_dim, output_dim, list_hidden_dims)

    self.apply(weight_init)
    self.temperature = torch.tensor(temperature)

  def forward(self, obs):
    logits = self.trunk(obs)
    dist = GumbelSoftmax(self.temperature, logits=logits)
    return dist

  def exploit(self, obs):
    dist = self.forward(obs)
    return dist.logits.argmax(dim=-1)

  def sample(self, obs):
    dist = self.forward(obs)

    samples = dist.sample()
    action_log_probs = dist.log_prob(samples).view(-1, 1)

    return samples, action_log_probs

  def rsample(self, obs):
    dist = self.forward(obs)

    action = dist.rsample()
    log_prob = dist.log_prob(action).view(-1, 1)

    return action, log_prob

  def is_discrete(self):
    return True
