import os
import pickle
import pygame
from tqdm import tqdm
from idil_gym.envs.cleanup_single import CleanupSingleEnv_v0
from hcair_domains.cleanup_single.mdp import MDPCleanupSingle
from hcair_domains.box_push import (BoxState, conv_box_state_2_idx,
                                    conv_box_idx_2_state)
import imageio
from ulid import ULID

def load_trajectories(file_path: str):
  with open(file_path, "rb") as f:
    data = pickle.load(f)
  return data


VISUALS_DIR = os.path.join("idil_train", "visuals")
TMP_DIR = os.path.join(VISUALS_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

def render_state(screen: pygame.Surface, env: CleanupSingleEnv_v0, 
                 state: int, tmp_dir : str= TMP_DIR, frame_idx: int = 0):
  mdp = env.mdp  # type: MDPCleanupSingle
  box_states, agent_pos = mdp.conv_mdp_sidx_to_sim_states(state)

  wid, hei = screen.get_size()
  x_unit = int(wid / mdp.x_grid)
  y_unit = int(hei / mdp.y_grid)

  clr_bg = (255, 255, 255)
  screen.fill(clr_bg)

  # draw walls
  for coord in mdp.walls:
    left = int(coord[0] * x_unit)
    top = int(coord[1] * y_unit)
    width, height = x_unit, y_unit
    pygame.draw.rect(screen, (0, 0, 0), (left, top, width, height))
  
  # draw goals
  for coord in mdp.goals:
    left = int(coord[0] * x_unit)
    top = int(coord[1] * y_unit)
    width, height = x_unit, y_unit
    pygame.draw.rect(screen, (255, 255, 0), (left, top, width, height))

  # draw original box places
  for coord in mdp.boxes:
    left = int(coord[0] * x_unit)
    top = int(coord[1] * y_unit)
    width, height = x_unit, y_unit
    pygame.draw.rect(screen, (200, 200, 200), (left, top, width, height))

  # draw boxes
  box_color = (195, 176, 138)
  with_agent = False
  for idx, bstateidx in enumerate(box_states):
    bstate = conv_box_idx_2_state(bstateidx, len(mdp.drops), len(mdp.goals))
    coord = None
    if bstate[0] == BoxState.Original:
      coord = mdp.boxes[idx]
      left = int((coord[0] + 0.2) * x_unit)
      top = int((coord[1] + 0.2) * y_unit)
      width, height = int(0.6 * x_unit), int(0.6 * y_unit)
    elif bstate[0] == BoxState.WithAgent1:
      with_agent = True
    elif bstate[0] == BoxState.OnGoalLoc:
      coord = mdp.goals[bstate[1]]
      left = int((coord[0] + 0.1 + 0.3 * idx) * x_unit)
      top = int((coord[1] + 0.1) * y_unit)
      width, height = int(0.25 * x_unit), int(0.25 * y_unit)
    else:
      raise ValueError("Invalid state")

    if coord is not None:
      pygame.draw.rect(screen, box_color, (left, top, width, height))

  # draw agent
  center = (int((agent_pos[0] + 0.5) * x_unit), 
            int((agent_pos[1] + 0.5) * y_unit))
  radius = int(x_unit * 0.3)
  pygame.draw.circle(screen, (0, 0, 255), center, radius)
  if with_agent:
    left = int((agent_pos[0] + 0.2) * x_unit)
    top = int((agent_pos[1] + 0.2) * y_unit)
    width, height = int(0.3 * x_unit), int(0.3 * y_unit)
    pygame.draw.rect(screen, box_color, (left, top, width, height))

  pygame.display.update()

  # save image of the current surface
  pygame.image.save(pygame.display.get_surface(), os.path.join(tmp_dir, f"state-{frame_idx}.png"))


def plot_trajectory(screen: pygame.Surface, env: CleanupSingleEnv_v0, states: list, traj_id: str= None):
  
  print("Generating state renders...\n")
  for idx, state in tqdm(enumerate(states)):
    render_state(screen, env, state=state, frame_idx=idx)
    pygame.time.wait(500)
  print("Generated successfully.\n")
  print("Creating gif...\n")
  # read all images in the tmp directory and create a gif
  frames = []
  for i in tqdm(range(len(states))):
    img_path = os.path.join(TMP_DIR, f"state-{i}.png")
    frames.append(imageio.imread(img_path))

  _id = str(ULID()) if not traj_id else traj_id
  imageio.mimsave(os.path.join(VISUALS_DIR, f"trajectory-{_id}.gif"), frames, duration=0.5)

  print("Saved successfully.")


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)
  file_path = os.path.join(cur_dir, "experts/CleanupSingle-v0_100.pkl")

  trajectories = load_trajectories(file_path)
  env = CleanupSingleEnv_v0()

  pygame.init()
  screen = pygame.display.set_mode((800, 800))

  n_epi = len(trajectories["states"])

  while True:
    traj_idx = input(f"Enter trajectory number (0~{n_epi-1}): ")
    if not traj_idx.isnumeric():
      print("Invalid input")
    else:
      traj_idx = int(traj_idx)
      if traj_idx < 0 or traj_idx >= n_epi:
        print("Out of the range")
      else:
        break

  states = trajectories["states"][traj_idx]
  len_traj = len(states)

  plot_trajectory(screen, env, states, traj_id=f"TRAJ_{str(traj_idx)}")