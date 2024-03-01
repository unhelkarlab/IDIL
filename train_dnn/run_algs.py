import os
import hydra
import datetime
import time
from omegaconf import OmegaConf, DictConfig
import gym_custom  # noqa: F401


def get_dirs(base_dir="",
             alg_name="gail",
             env_name="HalfCheetah-v2",
             msg="default"):

  base_log_dir = os.path.join(base_dir, "result/")

  ts_str = datetime.datetime.fromtimestamp(
      time.time()).strftime("%Y-%m-%d_%H-%M-%S")
  log_dir_root = os.path.join(base_log_dir, env_name, alg_name, msg, ts_str)
  save_dir = os.path.join(log_dir_root, "model")
  log_dir = os.path.join(log_dir_root, "log")
  os.makedirs(save_dir)
  os.makedirs(log_dir)

  return log_dir, save_dir


def run_alg(config):
  alg_name = config.alg_name
  msg = f"{config.tag}"
  log_interval = config.log_interval
  eval_interval = config.eval_interval

  log_dir, output_dir = get_dirs(config.base_dir, alg_name, config.env_name,
                                 msg)
  pretrain_name = os.path.join(config.base_dir, config.pretrain_path)
  # save config
  config_path = os.path.join(log_dir, "config.yaml")
  with open(config_path, "w") as outfile:
    OmegaConf.save(config=config, f=outfile)

  if (config.data_path.endswith("torch") or config.data_path.endswith("pt")
      or config.data_path.endswith("pkl") or config.data_path.endswith("npy")):
    path_iq_data = os.path.join(config.base_dir, config.data_path)
    num_traj = config.n_traj
  else:
    print(f"Data path not exists: {config.data_path}")

  if alg_name == "ogail":
    from ml_algs.baselines.option_gail.option_gail_learn import learn
    learn(config, log_dir, output_dir, path_iq_data, pretrain_name,
          eval_interval)
  elif alg_name == "idil_j":
    from ml_algs.IDIL_Joint.train import trainer
    trainer(config, path_iq_data, num_traj, log_dir, output_dir, log_interval,
            eval_interval)
  elif alg_name == "iql":
    from ml_algs.baselines.IQLearn.iql import run_iql
    run_iql(config, path_iq_data, num_traj, log_dir, output_dir, log_interval,
            eval_interval)
  elif alg_name == "sac":
    from ml_algs.baselines.IQLearn.iql import run_sac
    run_sac(config, log_dir, output_dir, log_interval, eval_interval)
  elif alg_name[:3] == "sb3":
    if alg_name[3:] == "bc":
      from sb3_algs import sb3_bc
      sb3_bc(config, path_iq_data, num_traj, log_dir, output_dir, log_interval)
  elif alg_name == "idil":
    from ml_algs.IDIL.train import train
    train(config, path_iq_data, num_traj, log_dir, output_dir, log_interval,
          eval_interval)
  else:
    raise ValueError("Invalid alg_name")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
  import torch.multiprocessing as multiprocessing
  multiprocessing.set_start_method('spawn')

  cur_dir = os.path.dirname(__file__)
  cfg.base_dir = cur_dir
  print(OmegaConf.to_yaml(cfg))

  run_alg(cfg)


if __name__ == "__main__":
  import time
  start_time = time.time()
  main()
  print("Excution time: ", time.time() - start_time)
