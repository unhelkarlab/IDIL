def get_possible_latent_states(num_boxes, num_drops, num_goals):
  latent_states = []
  for idx in range(num_boxes):
    latent_states.append(("pickup", idx))
  for idx in range(num_drops):
    latent_states.append(("drop", idx))
  for idx in range(num_goals):
    latent_states.append(("goal", idx))

  return latent_states
