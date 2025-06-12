import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from simple_ppo import PPO
from simple_ppo.policy import ContinuMlpPolicy
from simple_ppo.utils import plot

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--n-step", type=int, default=4096,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size for policy updates")
    parser.add_argument("--n-epochs", type=int, default=32,
        help="the number of epochs to update the policy")
    parser.add_argument("--clip-eps", type=float, default=0.1,
        help="the clipping epsilon for PPO")
    parser.add_argument("--vf-coef", type=float, default=1.0,
        help="the value function coefficient in the loss function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="the entropy coefficient in the loss function")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
        help="the maximum gradient norm")
    parser.add_argument("--eval-num", type=int, default=4,
        help="the number of evaluation episodes")

    args = parser.parse_args()
    # fmt: on
    return args

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)()
    env_eval = make_env(args.env_id, args.seed, 1, False, run_name)() # No video for eval env

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = ContinuMlpPolicy(state_dim=state_dim, action_dim=action_dim, sd_init=0.6, sd_rng=(0.01, 0.6)).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    agent = PPO(policy, optimizer, env, env_eval,
                gamma=args.gamma, gae_lambda=args.gae_lambda, n_step=args.n_step,
                batch_size=args.batch_size, n_epochs=args.n_epochs,
                clip_eps=args.clip_eps, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm, eval_num=args.eval_num, device=device)

    log = agent.train(total_timesteps=args.total_timesteps)

    # save log file for plotting
    import json
    log_path = f"runs/{run_name}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(f"{log_path}/log.json", 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {log_path}/log.json")

    env.close()
    env_eval.close()

    plot(log)
    print("Plot generated.")