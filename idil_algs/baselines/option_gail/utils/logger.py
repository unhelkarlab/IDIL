import os
from torch.utils.tensorboard import SummaryWriter


class Logger(object):

  def __init__(self, logdir=""):
    self.logdir = logdir

    if not os.path.exists(self.logdir):
      print(f"Making logging dir '{self.logdir}'")
      os.makedirs(self.logdir)
    self.writer = SummaryWriter(self.logdir)

  def log_loss(self, tag, v, i):
    self.writer.add_scalar(f"loss/{tag}", v, i)

  def log_loss_info(self, info_dict, i):
    for k in info_dict:
      self.writer.add_scalar(f"loss/{k}", info_dict[k], i)

  def log_train(self, tag, v, i):
    self.writer.add_scalar(f"train/{tag}", v, i)

  def log_train_fig(self, tag, fig, i):
    self.writer.add_figure(f"train/{tag}", fig, i, close=True)

  def log_train_info(self, info_dict, i):
    for k in info_dict:
      self.writer.add_scalar(f"train/{k}", info_dict[k], i)

  def log_test(self, tag, v, i):
    self.writer.add_scalar(f"test/{tag}", v, i)

  def log_test_fig(self, tag, fig, i):
    self.writer.add_figure(f"test/{tag}", fig, i, close=True)

  def log_test_info(self, info_dict, i):
    for k in info_dict:
      self.writer.add_scalar(f"test/{k}", info_dict[k], i)

  def log_eval(self, tag, v, i):
    self.writer.add_scalar(f"eval/{tag}", v, i)

  def log_eval_info(self, info_dict, i):
    for k in info_dict:
      self.writer.add_scalar(f"eval/{k}", info_dict[k], i)

  def log_pretrain(self, tag, v, i):
    self.writer.add_scalar(f"pretrain/{tag}", v, i)

  def log_pretrain_fig(self, tag, fig, i):
    self.writer.add_figure(f"pretrain/{tag}", fig, i, close=True)

  def log_pretrain_info(self, info_dict, i):
    for k in info_dict:
      self.writer.add_scalar(f"pretrain/{k}", info_dict[k], i)

  def flush(self):
    self.writer.flush()


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  logger = Logger("./log")
  plt.figure("123")
  for i in range(10):
    a = plt.figure("456")
    a.gca().plot(list(range(100)))
    logger.log_train_fig("cs", a, i)

  plt.plot(list(range(100, 0, -1)))
  plt.show()
