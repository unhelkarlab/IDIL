from collections import deque
import numpy as np
import random
import torch


class OptionMemory(object):

  def __init__(self,
               memory_size: int,
               seed: int = 0,
               use_deque: bool = True) -> None:
    random.seed(seed)
    self.memory_size = memory_size
    self.buffer = deque(maxlen=self.memory_size) if use_deque else list()

  def add(self, experience) -> None:
    'experience: obs, prev_lat, prev_act, next_obs, latent, action'
    self.buffer.append(experience)

  def size(self):
    return len(self.buffer)

  def sample(self, batch_size: int, continuous: bool = True):
    if batch_size > len(self.buffer):
      batch_size = len(self.buffer)
    if continuous:
      rand = random.randint(0, len(self.buffer) - batch_size)
      return [self.buffer[i] for i in range(rand, rand + batch_size)]
    else:
      indexes = np.random.choice(np.arange(len(self.buffer)),
                                 size=batch_size,
                                 replace=False)
      return [self.buffer[i] for i in indexes]

  def clear(self):
    self.buffer.clear()

  def save(self, path):
    b = np.asarray(self.buffer)
    print(b.shape)
    np.save(path, b)

  def get_all_samples(self, device):
    return self.get_samples(None, device=device)

  def get_samples(self, batch_size, device):
    if batch_size is None:
      batch = self.buffer
    else:
      batch = self.sample(batch_size, False)

    n_batch = len(batch)
    vec_batch_items = zip(*batch)

    list_batch_torch_items = []
    for batch_item in vec_batch_items:
      torch_item = torch.as_tensor(np.array(batch_item),
                                   dtype=torch.float,
                                   device=device).reshape(n_batch, -1)
      list_batch_torch_items.append(torch_item)

    return list_batch_torch_items
