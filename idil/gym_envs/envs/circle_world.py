import gym
from gym import spaces
import cv2
import numpy as np


class CircleWorld(gym.Env):
  # uncomment below line if you need to render the environment
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super().__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(low=-1,
                                   high=1,
                                   shape=(2, ),
                                   dtype=np.float32)

    self.half_sz = 5
    self.observation_space = spaces.Box(low=np.array([-self.half_sz, 0]),
                                        high=np.array(
                                            [self.half_sz, 2 * self.half_sz]),
                                        shape=(2, ),
                                        dtype=np.float32)

    self.reset()

  def step(self, action):
    info = {}

    scaler = 0.5

    next_obstate = self.cur_obstate + scaler * action
    next_obstate[0] = min(self.half_sz, max(-self.half_sz, next_obstate[0]))
    next_obstate[1] = min(2 * self.half_sz, max(0, next_obstate[1]))

    center = np.array([0, self.half_sz])
    dir = self.cur_obstate - center
    len_dir = np.linalg.norm(dir)
    if len_dir != 0:
      dir /= len_dir

    ortho = np.array([-dir[1], dir[0]])
    inner = ortho.dot(next_obstate - self.cur_obstate)
    reward = inner
    # reward = np.sign(inner) * (inner**2)

    self.cur_obstate = next_obstate

    return self.cur_obstate, reward, False, info

  def reset(self):
    degree = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0.8 * self.half_sz, self.half_sz)
    self.cur_obstate = (
        np.array([np.cos(degree), np.sin(degree)]) * radius +
        np.array([0, self.half_sz]))

    return self.cur_obstate

  def get_canvas(self):
    canvas_sz = 300

    canvas = np.ones((canvas_sz, canvas_sz, 3), dtype=np.uint8) * 255

    pt = np.array([self.cur_obstate[0] + self.half_sz, self.cur_obstate[1]])
    pt *= canvas_sz / (2 * self.half_sz)
    pt = pt.astype(np.int64)

    color = (255, 0, 0)
    canvas = cv2.circle(canvas, pt, 5, color, thickness=-1)

    return canvas

  def render(self, mode='human'):
    if mode == 'human':
      cv2.imshow("Circle World", self.get_canvas())
      cv2.waitKey(10)

  def close(self):
    cv2.destroyAllWindows()


if __name__ == "__main__":
  env = CircleWorld()

  # to see env.reset() works as intended
  canvas_sz = 300
  canvas = np.ones((canvas_sz, canvas_sz, 3), dtype=np.uint8) * 255

  for _ in range(100):
    state = env.reset()
    pt = np.array([state[0] + env.half_sz, state[1]])
    pt *= canvas_sz / (2 * env.half_sz)
    pt = pt.astype(np.int64)

    color = (255, 0, 0)
    canvas = cv2.circle(canvas, pt, 5, color, thickness=-1)
  cv2.imshow("Init States", canvas)
  cv2.waitKey(1000)
