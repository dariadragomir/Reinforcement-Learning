import os
import time
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

ENV_ID = "BreakoutNoFrameskip-v4"
RESULTS_DIR = "./proiect/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device(
   "cuda" if torch.cuda.is_available()
   else "mps" if torch.backends.mps.is_available()
   else "cpu"
)
print(f"[INFO] Using device: {device}")

torch.set_float32_matmul_precision("high")

GAMMA = 0.99
LR = 1e-4

BATCH_SIZE = 32
BUFFER_SIZE = 200_000
MIN_REPLAY_SIZE = 10_000

TARGET_TAU = 0.005
TRAIN_EVERY = 4

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_FRAMES = 1_000_000
MAX_FRAMES = 5_000_000
LOG_EVERY_FRAMES = 50_000

env = gym.make(ENV_ID)
env = AtariPreprocessing(
   env,
   grayscale_obs=True,
   scale_obs=False,
   frame_skip=4,
   screen_size=84
)
env = FrameStackObservation(env, 4)

n_actions = env.action_space.n

class ReplayBuffer:
   def __init__(self, capacity):
       self.buffer = deque(maxlen=capacity)

   def push(self, s, a, r, s2, d):
       self.buffer.append((s, a, r, s2, d))

   def sample(self, batch_size):
       batch = random.sample(self.buffer, batch_size)
       s, a, r, s2, d = map(np.array, zip(*batch))
       return s, a, r, s2, d

   def __len__(self):
       return len(self.buffer)

class DQN(nn.Module):
   def __init__(self, n_actions):
       super().__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(4, 32, 8, stride=4),
           nn.ReLU(),
           nn.Conv2d(32, 64, 4, stride=2),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, stride=1),
           nn.ReLU(),
       )
       self.fc = nn.Sequential(
           nn.Linear(7 * 7 * 64, 512),
           nn.ReLU(),
           nn.Linear(512, n_actions)
       )

   def forward(self, x):
       x = x / 255.0
       x = self.conv(x)
       x = x.view(x.size(0), -1)
       return self.fc(x)

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "mps"))

replay = ReplayBuffer(BUFFER_SIZE)

def epsilon_by_frame(frame):
   return EPS_END + (EPS_START - EPS_END) * np.exp(-frame / EPS_DECAY_FRAMES)

def select_action(state, eps):
   if random.random() < eps:
       return env.action_space.sample()
   with torch.no_grad():
       state = torch.as_tensor(state, device=device).unsqueeze(0)
       return policy_net(state).argmax(1).item()

episode_rewards = []
losses = []
q_values_mean = []
fps_history = []
state, _ = env.reset()
episode_reward = 0

frame_count = 0
start_time = time.time()
last_log_time = start_time

print("[INFO] Starting training...")

while frame_count < MAX_FRAMES:
   eps = epsilon_by_frame(frame_count)
   action = select_action(state, eps)

   next_state, reward, terminated, truncated, _ = env.step(action)
   done = terminated or truncated

   replay.push(state, action, reward, next_state, done)
   state = next_state
   episode_reward += reward
   frame_count += 1

   if done:
       episode_rewards.append(episode_reward)
       state, _ = env.reset()
       episode_reward = 0

   if len(replay) < MIN_REPLAY_SIZE or frame_count % TRAIN_EVERY != 0:
       continue

   s, a, r, s2, d = replay.sample(BATCH_SIZE)

   s = torch.as_tensor(s, device=device)
   a = torch.as_tensor(a, device=device).unsqueeze(1)
   r = torch.as_tensor(r, device=device, dtype=torch.float32).unsqueeze(1)
   s2 = torch.as_tensor(s2, device=device)
   d = torch.as_tensor(d, device=device, dtype=torch.float32).unsqueeze(1)

   with torch.autocast(device_type=device.type, dtype=torch.float16):
       q = policy_net(s).gather(1, a)
       with torch.no_grad():
           q_next = target_net(s2).max(1, keepdim=True)[0]
           q_target = r + GAMMA * (1 - d) * q_next
       loss = F.smooth_l1_loss(q, q_target)

   optimizer.zero_grad()
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()

   losses.append(loss.item())
   q_values_mean.append(q.mean().item())

   for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
       tp.data.copy_(TARGET_TAU * pp.data + (1 - TARGET_TAU) * tp.data)

   if frame_count % LOG_EVERY_FRAMES == 0:
       now = time.time()
       fps = LOG_EVERY_FRAMES / (now - last_log_time)
       last_log_time = now
       fps_history.append(fps)

       print(
           f"[STEP {frame_count:,}] "
           f"Episodes: {len(episode_rewards)} | "
           f"Replay: {len(replay)} | "
           f"ε={eps:.3f} | "
           f"FPS={fps:.1f}"
       )

print("[INFO] Training complete")

def save_plot(data, title, ylabel, filename, moving_avg=None):
   plt.figure(figsize=(8, 4))
   plt.plot(data, label="raw")
   if moving_avg is not None and len(data) >= moving_avg:
       avg = np.convolve(data, np.ones(moving_avg)/moving_avg, mode="valid")
       plt.plot(range(moving_avg-1, moving_avg-1+len(avg)), avg, label="avg")
   plt.title(title)
   plt.xlabel("Steps")
   plt.ylabel(ylabel)
   plt.legend()
   plt.tight_layout()
   plt.savefig(os.path.join(RESULTS_DIR, filename))
   plt.close()

save_plot(episode_rewards, "Episode Rewards", "Reward", "rewards.png", moving_avg=50)
save_plot(losses, "Training Loss", "Loss", "loss.png")
save_plot(q_values_mean, "Mean Q-Values", "Q", "q_values.png")
save_plot(fps_history, "Performance (FPS)", "FPS", "performance.png")

def plot_combined_metrics(rewards, losses, q_values, fps, save_dir):
   plt.style.use('seaborn-v0_8-darkgrid')
  
   fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=200)
   fig.suptitle(f'DQN Training Metrics: {ENV_ID}', fontsize=16, weight='bold')

   def moving_average(data, window_size):
       if len(data) < window_size:
           return data
       return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

   # plot 1: reward
   axs[0, 0].plot(rewards, alpha=0.2, color='gray', label='Raw')
  
   rew_window = max(1, int(len(rewards) * 0.05))
   avg_rewards = moving_average(rewards, window_size=50)
  
   x_avg = range(len(rewards) - len(avg_rewards), len(rewards))
   axs[0, 0].plot(x_avg, avg_rewards, color='#1f77b4', linewidth=2, label='Moving Avg (50)')
  
   axs[0, 0].set_title('Episode Rewards', fontsize=12, weight='bold')
   axs[0, 0].set_ylabel('Reward')
   axs[0, 0].set_xlabel('Episodes')
   axs[0, 0].legend(loc='upper left')


   # plot 2: loss 
   loss_window = max(1, int(len(losses) * 0.01))
   avg_loss = moving_average(losses, window_size=loss_window)
   x_loss = np.linspace(0, len(losses), len(avg_loss))
  
   axs[0, 1].plot(x_loss, avg_loss, color='#d62728', linewidth=1.5)
   axs[0, 1].set_title(f'Training Loss (Smoothed w/{loss_window})', fontsize=12, weight='bold')
   axs[0, 1].set_ylabel('Loss')
   axs[0, 1].set_xlabel('Training Steps')


   # plot 3: Q-Values
   q_window = max(1, int(len(q_values) * 0.01))
   avg_q = moving_average(q_values, window_size=q_window)
   x_q = np.linspace(0, len(q_values), len(avg_q))
  
   axs[1, 0].plot(x_q, avg_q, color='#2ca02c', linewidth=1.5)
   axs[1, 0].set_title(f'Mean Q-Values (Smoothed w/{q_window})', fontsize=12, weight='bold')
   axs[1, 0].set_ylabel('Q-Value')
   axs[1, 0].set_xlabel('Training Steps')


   # plot 4: FPS (CORRECTED) ---
   x_fps = np.arange(1, len(fps) + 1) * LOG_EVERY_FRAMES
  
   axs[1, 1].plot(x_fps, fps, color='#9467bd', linewidth=2, marker='o', markersize=3)
   axs[1, 1].set_title('Training Speed (FPS)', fontsize=12, weight='bold')
   axs[1, 1].set_ylabel('FPS')
   axs[1, 1].set_xlabel('Total Steps')

   plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  
   save_path = os.path.join(save_dir, "training_summary.png")
   plt.savefig(save_path)
   print(f"[INFO] Combined plot saved to: {save_path}")
   plt.close()

plot_combined_metrics(episode_rewards, losses, q_values_mean, fps_history, RESULTS_DIR)

print(f"[INFO] Plots saved in ./{RESULTS_DIR}")
