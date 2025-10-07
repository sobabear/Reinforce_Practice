"""
Some miscellaneous utility functions

Functions to edit:
    1. rollout_trajectory (line 19)
    2. rollout_trajectories (line 67)
    3. rollout_n_trajectories (line 83)
"""

import numpy as np
import time

############################################
############################################

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
MJ_ENV_KWARGS["Ant-v4"]["use_contact_forces"] = True


def rollout_trajectory(env, policy, max_traj_length, render=False):
    """
    Rolls out a policy and generates a trajectories

    :param policy: the policy to roll out
    :param max_traj_length: the number of steps to roll out
    :render: whether to save images from the rollout
    """
    # initialize env for the beginning of a new rollout
    # TODO: implement the following line
    ob, _ = None, None  # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if hasattr(env, "sim"):
                image_obs.append(
                    env.sim.render(camera_name="track", height=500, width=500)[::-1]
                )
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = None  # HINT: query the policy's get_action function
        acs.append(ac)

        # take that action and record results
        ob, rew, terminated, truncated, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to termination or truncation, or due to exceeding or reaching (>=) max_traj_length
        rollout_done = (None or None) or None  # HINT: this is either 0 or 1

        terminals.append(rollout_done)

        if rollout_done:
            break

    return Traj(obs, image_obs, acs, rewards, next_obs, terminals)


def rollout_trajectories(
    env, policy, min_timesteps_per_batch, max_traj_length, render=False
):
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.

    TODO implement this function
    Hint1: use `rollout_trajectory` to get each traj (i.e. rollout) that goes into trajs
    Hint2: use `get_trajlength` to count the timesteps collected in each traj
    Hint3: repeat while we have collected at least min_timesteps_per_batch steps
    """
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        pass

    return trajs, timesteps_this_batch


def rollout_n_trajectories(env, policy, ntraj, max_traj_length, render=False):
    """
    Collect ntraj rollouts.

    TODO implement this function
    Hint1: use rollout_trajectory to get each traj (i.e. rollout) that goes into trajs
    """
    trajs = []

    return trajs


############################################
############################################


def Traj(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def convert_listofrollouts(trajs, concat_rew=True):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    if concat_rew:
        rewards = np.concatenate([traj["reward"] for traj in trajs])
    else:
        rewards = [traj["reward"] for traj in trajs]
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    return observations, actions, rewards, next_observations, terminals


############################################
############################################


def get_trajlength(traj):
    return len(traj["reward"])
