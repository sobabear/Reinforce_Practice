"""
Defines a trainer which updates a behavioral cloning agent

Functions to edit:
    1. collect_training_trajectories (line 180)
    2. train_agent line(223)
    3. do_relabel_with_expert (line 243)
"""

from collections import OrderedDict

import pickle
import time
import torch
import gymnasium as gym

import numpy as np

from gcb6206.infrastructure import pytorch_util as ptu
from gcb6206.infrastructure.logger import Logger
from gcb6206.infrastructure import utils

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class BCTrainer:
    """
    A class which defines the training algorithm for the agent. Handles
    sampling data, updating the agent, and logging the results.

    ...

    Attributes
    ----------
    agent : BCAgent
        The agent we want to train

    Methods
    -------
    run_training_loop:
        Main training loop for the agent
    collect_training_trajectories:
        Collect data to be used for training
    train_agent
        Samples a batch and updates the agent
    do_relabel_with_expert
        Relabels trajectories with new actions for DAgger
    """

    def __init__(self, params):
        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params["logdir"])

        # Set random seeds
        seed = self.params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(use_gpu=not self.params["no_gpu"], gpu_id=self.params["which_gpu"])

        # Set logger attributes
        self.log_video = True
        self.log_metrics = True

        #############
        ## ENV
        #############

        # Make the gym environment
        if self.params["video_log_freq"] == -1:
            self.params["env_kwargs"]["render_mode"] = None
        self.env = gym.make(self.params["env_name"], **self.params["env_kwargs"])
        self.env.reset(seed=seed)

        # Maximum length for episodes
        self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params["ep_len"]

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params["agent_params"]["discrete"] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim

        # simulation timestep, will be used for video saving
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata["render_fps"]

        #############
        ## AGENT
        #############

        agent_class = self.params["agent_class"]
        self.agent = agent_class(self.env, self.params["agent_params"])

    def run_training_loop(
        self,
        n_iter,
        collect_policy,
        eval_policy,
        initial_expertdata=None,
        relabel_with_expert=False,
        start_relabel_with_expert=1,
        expert_policy=None,
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)

            # decide if videos should be rendered/logged at this iteration
            if (
                itr % self.params["video_log_freq"] == 0
                and self.params["video_log_freq"] != -1
            ):
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.params["scalar_log_freq"] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr, collect_policy, initial_expertdata
            )  # HW1: implement this function below
            trajs, envsteps_this_batch, train_video_trajs = training_returns
            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                # HW1: implement this function below
                trajs = self.do_relabel_with_expert(expert_policy, trajs)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(trajs)

            # train agent (using sampled data from replay buffer)
            # HW1: implement this function below
            training_logs = self.train_agent()

            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print("\nBeginning logging procedure...")
                self.perform_logging(
                    itr, trajs, eval_policy, train_video_trajs, training_logs
                )

                if self.params["save_params"]:
                    print("\nSaving agent params")
                    self.agent.save(
                        "{}/policy_itr_{}.pt".format(self.params["logdir"], itr)
                    )

    ####################################
    ####################################

    def collect_training_trajectories(
        self, itr, collect_policy, load_initial_expertdata=None
    ):
        """
        :param itr:
        :param load_initial_expertdata: path to expert data pkl file
        :param collect_policy: the current policy using which we collect data
        :return:
            trajs: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in trajs
            train_video_trajs: trajs which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT1: depending on if it's the first iteration or not, decide whether to either
        # (1) If it is the first iteration and training data exists, load it using pickle.load.
        # In this case, you can directly return as follows
        # ``` return loaded_trajs, 0, None ```
        # (2) If no training data exists,
        # collect `self.params['batch_size_initial']` transitions
        # (3) If it is not the first iteration (the case of DAgger implementation),
        # collect `self.params['batch_size']` transitions
        # HINT2: use rollout_trajectories from utils (after the "Collecting data .." print statement).
        # HINT3: you want each of these collected rollouts to be of length self.params['ep_len']

        print("\nCollecting data to be used for training...")
        trajs, envsteps_this_batch = None, None
        ## TODO: implement the above logic

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_trajs = None
        if self.log_video:
            print("\nCollecting train rollouts to be used for saving videos...")
            ## TODO look in utils and implement rollout_n_trajectories

        return trajs, envsteps_this_batch, train_video_trajs

    def train_agent(self):
        """
        Samples a batch of trajectories and updates the agent with the batch
        """
        print("\nTraining agent using sampled data from replay buffer...")
        all_logs = []
        for train_step in range(self.params["num_agent_train_steps_per_iter"]):
            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = (
                None,
                None,
                None,
                None,
                None,
            )

            # TODO use the sampled data to train an agent
            # HINT3: use the agent's train function
            # HINT4: keep the agent's training log for debugging
            train_log = None
            all_logs.append(train_log)
        return all_logs

    def do_relabel_with_expert(self, expert_policy, trajs):
        """
        Relabels collected trajectories with an expert policy

        :param expert_policy: the policy we want to relabel the trajs with
        :param trajs: trajs to relabel
        """
        print(
            "\nRelabelling collected observations with labels from an expert policy..."
        )

        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with trajs[i]["observation"]
        # and replace trajs[i]["action"] with these expert labels

        return trajs

    ####################################
    ####################################

    def perform_logging(
        self, itr, trajs, eval_policy, train_video_trajs, training_logs
    ):
        """
        Logs training trajectories and evals the provided policy to log
        evaluation trajectories and videos

        :param itr:
        :param trajs: trajs collected during training that we want to log
        :param eval_policy: policy to generate eval logs and videos
        :param train_video_trajs: videos generated during training
        :param training_logs: additional logs generated during training
        """

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_trajs, eval_envsteps_this_batch = utils.rollout_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], self.params["ep_len"]
        )

        # save eval rollouts as videos in tensorboard event file
        if self.log_video:
            print("\nCollecting video rollouts eval")
            eval_video_trajs = utils.rollout_n_trajectories(
                self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )
        else:
            eval_video_trajs = None

        # save train/eval videos
        print("\nSaving rollouts as videos...")
        if train_video_trajs is not None:
            self.logger.log_trajs_as_videos(
                train_video_trajs,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="train_rollouts",
            )
        if eval_video_trajs is not None:
            self.logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [traj["reward"].sum() for traj in trajs]
            eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

            # episode lengths, for logging
            train_ep_lens = [len(traj["reward"]) for traj in trajs]
            eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()
