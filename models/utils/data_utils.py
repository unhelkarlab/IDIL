from typing import List
import numpy as np
import random


class Trajectories:
  EPISODE_END = -1
  DUMMY = -2

  def __init__(self,
               num_state_factors,
               num_action_factors,
               num_latent_factors=0,
               tup_num_latents=(0, 0)) -> None:
    self.list_np_trajectory = []  # type: List[np.ndarray]
    self.tup_num_latents = tup_num_latents
    self.num_samples_to_use = None

    # TODO: make more generic
    self.num_state_factors = num_state_factors
    self.num_action_factors = num_action_factors
    self.num_latent_factors = num_latent_factors

  def shuffle(self):
    if len(self.list_np_trajectory) == 0:
      return

    random.shuffle(self.list_np_trajectory)

  def get_num_valid_rows(self):
    count = 0
    for traj in self.list_np_trajectory:
      count += traj.shape[0]
      if self._is_episode_end(traj[-1]):
        count -= 1

    return count

  def set_num_samples_to_use(self, num_sample):
    self.num_samples_to_use = num_sample

  def get_num_samples_to_use(self):
    return (len(self.list_np_trajectory)
            if self.num_samples_to_use is None else self.num_samples_to_use)

  def load_from_files(self, file_names):
    raise NotImplementedError

  def get_width(self):
    return (self.num_state_factors + self.num_action_factors +
            self.num_latent_factors)

  def _is_episode_end(self, np_row):
    '''
    can be the terminal of episode or
    just a normal state in case episode ended early due to max step count
    '''
    return np_row[self.num_state_factors] == Trajectories.EPISODE_END

  def get_as_row_lists(self, no_latent_label: bool, include_terminal: bool):
    list_list_SAX = []
    for np_trj in self.list_np_trajectory[:self.get_num_samples_to_use()]:
      list_SAX = []
      for idx in range(np_trj.shape[0]):
        list_tmp = []
        # TODO: make more generic
        # state
        start_idx, end_idx = 0, self.num_state_factors
        if self.num_state_factors == 1:
          list_tmp.append(np_trj[idx][start_idx:end_idx][0])
        else:
          list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))

        if self._is_episode_end(np_trj[idx]):
          if include_terminal:
            list_tmp.append(None)  # action
            if self.num_latent_factors > 0:
              list_tmp.append(None)  # latent
          else:
            break
        else:
          # action
          start_idx, end_idx = end_idx, end_idx + self.num_action_factors
          if self.num_action_factors == 1:
            list_tmp.append(np_trj[idx][start_idx:end_idx][0])
          else:
            list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))

          # latent
          if self.num_latent_factors > 0:
            if no_latent_label:
              if self.num_latent_factors == 1:
                list_tmp.append(None)
              else:
                list_tmp.append((None, ) * self.num_latent_factors)
            else:
              start_idx, end_idx = end_idx, end_idx + self.num_latent_factors
              if self.num_latent_factors == 1:
                list_tmp.append(np_trj[idx][start_idx:end_idx][0])
              else:
                list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))

        list_SAX.append(list_tmp)
      list_list_SAX.append(list_SAX)
    return list_list_SAX

  def get_as_column_lists(self, include_terminal: bool):
    list_trajectories = []
    for np_trj in self.list_np_trajectory[:self.get_num_samples_to_use()]:
      is_terminal = self._is_episode_end(np_trj[-1, :])
      list_traj = []
      row_end_idx = (-1 if is_terminal and not include_terminal else
                     np_trj.shape[0])
      start_idx, end_idx = 0, self.num_state_factors
      if self.num_state_factors == 1:
        list_states = list(np_trj[:row_end_idx,
                                  start_idx:end_idx].squeeze(axis=1))
      else:
        list_states = list(map(tuple, np_trj[:row_end_idx, start_idx:end_idx]))
      list_traj.append(list_states)

      row_end_idx = -1 if is_terminal else np_trj.shape[0]
      start_idx, end_idx = end_idx, end_idx + self.num_action_factors
      if self.num_action_factors == 1:
        list_actions = list(np_trj[:row_end_idx,
                                   start_idx:end_idx].squeeze(axis=1))
      else:
        list_actions = list(map(tuple, np_trj[:row_end_idx, start_idx:end_idx]))
      list_traj.append(list_actions)

      if self.num_latent_factors > 0:
        start_idx, end_idx = end_idx, end_idx + self.num_latent_factors
        if self.num_latent_factors == 1:
          list_latents = list(np_trj[:row_end_idx,
                                     start_idx:end_idx].squeeze(axis=1))
        else:
          list_latents = list(
              map(tuple, np_trj[:row_end_idx, start_idx:end_idx]))
        list_traj.append(list_latents)

      list_trajectories.append(list_traj)

    return list_trajectories

  def get_trajectories_fragmented_by_latent(self, include_next_state: bool):
    '''
    num_samples_to_use: set the number of trajectories to use for training
                          if None, all data are assumed to be used
    return: trajectories fragmented by latents for each agent
            can be accessed as list_name[agent][latent][trajectory][time_step]
    only works when num_action_factors == num_latent_factors
    '''
    assert self.num_action_factors == self.num_latent_factors

    list_by_agent = []
    idx_lat_start = self.num_state_factors + self.num_action_factors
    for nidx in range(self.num_action_factors):
      list_lat = [list() for dummy in range(self.tup_num_latents[nidx])]
      for np_trj in self.list_np_trajectory[:self.get_num_samples_to_use()]:
        if np_trj.shape[0] < 2:
          continue

        list_frag_trj = []
        for tidx in range(np_trj.shape[0] - 1):
          tidx_n = tidx + 1

          np_sidx = np_trj[tidx, 0:self.num_state_factors]
          sidx = np_sidx[0] if self.num_state_factors == 1 else tuple(np_sidx)

          np_sidx_n = np_trj[tidx_n, 0:self.num_state_factors]
          sidx_n = (np_sidx_n[0]
                    if self.num_state_factors == 1 else tuple(np_sidx_n))

          aidx = np_trj[tidx, self.num_state_factors + nidx]
          aidx_n = np_trj[tidx_n, self.num_state_factors + nidx]

          xidx = np_trj[tidx, idx_lat_start + nidx]
          xidx_n = np_trj[tidx_n, idx_lat_start + nidx]

          list_frag_trj.append((sidx, aidx))
          if xidx != xidx_n:
            if include_next_state:
              if self._is_episode_end(np_trj[tidx_n]):
                list_frag_trj.append((sidx_n, Trajectories.EPISODE_END))
              else:
                list_frag_trj.append((sidx_n, Trajectories.DUMMY))

            list_lat[xidx].append(list_frag_trj)
            list_frag_trj = []
          # handle the last element if not handled yet
          # this cannot be terminal as the case is already handled above
          elif tidx == np_trj.shape[0] - 2:
            if include_next_state:
              list_frag_trj.append((sidx_n, Trajectories.DUMMY))
            else:
              list_frag_trj.append((sidx_n, aidx_n))
            list_lat[xidx].append(list_frag_trj)

      list_by_agent.append(list_lat)
    return list_by_agent

  def get_as_row_lists_for_static_x(self, include_terminal: bool = False):
    list_list_SA = []
    list_X = []
    for np_trj in self.list_np_trajectory[:self.get_num_samples_to_use()]:
      list_SA = []
      for idx in range(np_trj.shape[0]):
        list_tmp = []
        # TODO: make more generic
        # state
        start_idx, end_idx = 0, self.num_state_factors
        if self.num_state_factors == 1:
          list_tmp.append(np_trj[idx][start_idx:end_idx][0])
        else:
          list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))

        if self._is_episode_end(np_trj[idx]):
          if include_terminal:
            list_tmp.append(None)  # action
          else:
            break
        else:
          # action
          start_idx, end_idx = end_idx, end_idx + self.num_action_factors
          if self.num_action_factors == 1:
            list_tmp.append(np_trj[idx][start_idx:end_idx][0])
          else:
            list_tmp.append(tuple(np_trj[idx][start_idx:end_idx]))

        list_SA.append(list_tmp)

      # latent
      if self.num_latent_factors > 0:
        start_idx = self.num_state_factors + self.num_action_factors
        end_idx = (self.num_state_factors + self.num_action_factors +
                   self.num_latent_factors)

        if self.num_latent_factors == 1:
          list_X.append(np_trj[0][start_idx:end_idx][0])
        else:
          list_X.append(tuple(np_trj[0][start_idx:end_idx]))

      list_list_SA.append(list_SA)
    return list_list_SA, list_X
