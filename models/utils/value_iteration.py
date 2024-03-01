from typing import Optional, Union

import numpy as np
from tqdm import tqdm
import sparse

import models.mdp as mdp_lib


def value_iteration(
    transition_model: Union[np.ndarray, sparse.COO],
    reward_model: np.ndarray,
    discount_factor: float = 0.95,
    max_iteration: int = 20,
    epsilon: float = 1e-6,
    v_value_initial: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Implements the value iteration algorithm.

  Args:
    transition_model: A transition model as a numpy 3-d array.
    reward_model: A reward model as a numpy 2-d array.
    discount_factor: MDP discount factor to be used for policy evaluation.
    max_iteration: Maximum number of iterations for policy evaluation.
    epsilon: Desired v-value threshold. Used for termination condition.
    v_value_initial: Optional. Initial guess for V value.

  Returns:
    A tuple of policy, v_value, and q_value.
  """
  num_states, num_actions, _ = transition_model.shape

  if v_value_initial is not None:
    assert v_value_initial.shape == (num_states, ), (
        "Initial V value has incorrect shape.")
    v_value = v_value_initial
  else:
    v_value = np.zeros((num_states))

  iteration_idx = 0
  delta_v = epsilon + 1.
  progress_bar = tqdm(total=max_iteration)
  while (iteration_idx < max_iteration) and (delta_v > epsilon):
    q_value = mdp_lib.q_value_from_v_value(v_value, transition_model,
                                           reward_model, discount_factor)
    new_v_value = q_value.max(axis=-1)
    delta_v = np.linalg.norm(
        np.nan_to_num(new_v_value[:]) - np.nan_to_num(v_value[:]))
    iteration_idx += 1
    v_value = new_v_value
    progress_bar.set_postfix({'delta': delta_v})
    progress_bar.update()
  progress_bar.close()

  policy = mdp_lib.deterministic_policy_from_q_value(q_value)

  return (policy, v_value, q_value)
