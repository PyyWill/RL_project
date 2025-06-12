import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

class DQN:
    def __init__(
        self,
        args,
        envs,
        q_network,
        target_network,
        optimizer,
        device,
        writer,
        run_name
    ):
        self.args = args
        self.envs = envs
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.run_name = run_name

    def train(self, total_timesteps):
        rb = ReplayBuffer(
            self.args.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        obs, _ = self.envs.reset(seed=self.args.seed)
        for global_step in range(total_timesteps):
            epsilon = linear_schedule(self.args.start_e, self.args.end_e, self.args.exploration_fraction * total_timesteps, global_step)
            if random.random() < epsilon:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                q_values = self.q_network(torch.Tensor(obs).to(self.device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        self.writer.add_scalar("charts/epsilon", epsilon, global_step)

            real_next_obs = next_obs.copy()
            for idx, d in enumerate(truncated):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

            obs = next_obs

            if global_step > self.args.learning_starts:
                if global_step % self.args.train_frequency == 0:
                    data = rb.sample(self.args.batch_size)
                    with torch.no_grad():
                        target_max, _ = self.target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + self.args.gamma * target_max * (1 - data.dones.flatten())
                    old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        self.writer.add_scalar("losses/td_loss", loss, global_step)
                        self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if global_step % self.args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_network_param.data.copy_(
                            self.args.tau * q_network_param.data + (1.0 - self.args.tau) * q_network_param.data
                        )

        if self.args.save_model:
            model_path = f"runs/{self.run_name}/{self.args.exp_name}.pth"
            torch.save(self.q_network.state_dict(), model_path)
            print(f"model saved to {model_path}")

        self.envs.close()
        self.writer.close() 