## Setup

We recommend you to use our pre-configured docker image `luminoctua/gcb6206:latest`, which already has all necessary packages for this homework. If you want to use your own computer, see [installation.md](installation.md). If you want to use Google Colab, see [colab_instructions.md](colab_instructions.md).

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/bc_trainer.py](gcb6206/infrastructure/bc_trainer.py)
 - [policies/MLP_policy.py](gcb6206/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](gcb6206/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](gcb6206/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](gcb6206/infrastructure/pytorch_util.py)

Look for sections marked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_hw1.py](gcb6206/scripts/run_hw1.py)
 - [agents/bc_agent.py](gcb6206/agents/bc_agent.py)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!


### Section 1 (Behavioral Cloning)
Command for problem 1:

```
python gcb6206/scripts/run_hw1.py \
	--expert_policy_file gcb6206/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data gcb6206/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python gcb6206/scripts/run_hw1.py \
    --expert_policy_file gcb6206/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data gcb6206/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```
