import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

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

ENV_ID = "BreakoutNoFrameskip-v4"
RESULTS_DIR = "./c51_breakout"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"[INFO] Using device: {device}")
torch.set_float32_matmul_precision("high")


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

# C51 Network Distributional
class C51(nn.Module):
    def __init__(self, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        
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
            nn.Linear(512, n_actions * n_atoms) 
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        logits = logits.view(-1, self.n_actions, self.n_atoms)
        
        return F.log_softmax(logits, dim=-1)

    def get_q_value(self, x):
        log_probs = self(x)
        probs = log_probs.exp()
        q_values = (probs * self.atoms).sum(dim=2)
        return q_values

def epsilon_by_frame(frame, eps_start, eps_end, eps_decay):
    return eps_end + (eps_start - eps_end) * np.exp(-frame / eps_decay)

def select_action(env, network, state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    
    with torch.no_grad():
        state_t = torch.as_tensor(state, device=device).unsqueeze(0)
        #we select action with highest expected q-value
        q_values = network.get_q_value(state_t) 
        return q_values.argmax(1).item()

def plot_combined_metrics(rewards, losses, fps, save_dir, exp_idx):
    plt.style.use('seaborn-v0_8-darkgrid') 
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=200)
    fig.suptitle(f'C51 Training Metrics', fontsize=16, weight='bold')

    def moving_average(data, window_size):
        if len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    axs[0, 0].plot(rewards, alpha=0.2, color='gray')
    avg_rewards = moving_average(rewards, 50) 
    if len(avg_rewards) > 0:
        axs[0, 0].plot(range(len(rewards)-len(avg_rewards), len(rewards)), avg_rewards, color='#1f77b4')
    axs[0, 0].set_title('Episode Rewards')

    axs[0, 1].plot(losses, alpha=0.3, color='#d62728')
    avg_loss = moving_average(losses, 100)
    if len(avg_loss) > 0:
        axs[0, 1].plot(range(len(losses)-len(avg_loss), len(losses)), avg_loss, color='darkred')
    axs[0, 1].set_title('Loss')

    axs[1, 1].plot(fps, color='#9467bd')
    axs[1, 1].set_title('FPS')

    plt.tight_layout() 
    plt.savefig(os.path.join(save_dir, f"training_summary_{exp_idx}.png"))
    plt.close()


V_MIN = -10.0
V_MAX = 10.0
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)

EXP_NUM = 0
configs = [(1e-4, 64, 1_000_000, 0.01, 100_000)]

for lr, bs, eps_decay, eps_end, buffer_size in configs:
    GAMMA = 0.99
    LR = lr
    BATCH_SIZE = bs
    MIN_REPLAY_SIZE = 10_000
    TARGET_TAU = 0.005 
    TRAIN_EVERY = 4
    MAX_FRAMES = 5_000_000
    LOG_EVERY_FRAMES = 50_000

  
    env = gym.make(ENV_ID)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4, screen_size=84, terminal_on_life_loss=True)
    env = FrameStackObservation(env, 4)
    n_actions = env.action_space.n


    policy_net = C51(n_actions, N_ATOMS, V_MIN, V_MAX).to(device)
    target_net = C51(n_actions, N_ATOMS, V_MIN, V_MAX).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR, eps=1e-5) # eps needed for stability in C51
    replay = ReplayBuffer(buffer_size)

    
    episode_rewards = []
    losses = []
    fps_history = []
    
    state, _ = env.reset()
    episode_reward = 0
    frame_count = 0
    start_time = time.time()
    last_log_time = start_time

    print(f"[INFO] Starting C51 Training (Exp {EXP_NUM})...")

    while frame_count < MAX_FRAMES:
        eps = epsilon_by_frame(frame_count, 1.0, eps_end, eps_decay)
        action = select_action(env, policy_net, state, eps)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        #clip rewards to [-1, 1] to stay within support range
        reward = np.clip(reward, -1.0, 1.0)

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
        a = torch.as_tensor(a, device=device).long() 
        r = torch.as_tensor(r, device=device, dtype=torch.float32)
        s2 = torch.as_tensor(s2, device=device)
        d = torch.as_tensor(d, device=device, dtype=torch.float32)

        with torch.no_grad():
            next_log_probs = target_net(s2)
            next_probs = next_log_probs.exp()

          
            next_q_values = (next_probs * target_net.atoms).sum(dim=2)
            next_actions = next_q_values.argmax(1) # (B,)


            next_dist = next_probs[range(BATCH_SIZE), next_actions]

            t_z = r.unsqueeze(1) + GAMMA * (1 - d.unsqueeze(1)) * target_net.atoms.unsqueeze(0)
            t_z = t_z.clamp(min=V_MIN, max=V_MAX)
            
            b = (t_z - V_MIN) / DELTA_Z
            l = b.floor().long()
            u = b.ceil().long()

          
            l[(u > 0) * (l == u)] -= 1
            j = torch.linspace(0, (BATCH_SIZE - 1) * N_ATOMS, BATCH_SIZE, device=device).long().unsqueeze(1)
            
            
            proj_dist = torch.zeros((BATCH_SIZE, N_ATOMS), device=device)
            
            
            proj_dist.view(-1).index_add_(0, (l + j).view(-1), (next_dist * (u.float() - b)).view(-1))
            
            # Mass for upper index: next_dist * (b - l)
            proj_dist.view(-1).index_add_(0, (u + j).view(-1), (next_dist * (b - l.float())).view(-1))

        # cross entropy loss: -Sum(Target * Log(Prediction))
        
        current_log_probs = policy_net(s) 
        
        current_log_probs_action = current_log_probs[range(BATCH_SIZE), a] # (B, Atoms)
        

        loss = - (proj_dist * current_log_probs_action).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.copy_(TARGET_TAU * pp.data + (1 - TARGET_TAU) * tp.data)

        if frame_count % LOG_EVERY_FRAMES == 0:
            now = time.time()
            fps = LOG_EVERY_FRAMES / (now - last_log_time)
            last_log_time = now
            fps_history.append(fps)
            print(f"[STEP {frame_count:,}] Rewards: {np.mean(episode_rewards[-20:]):.2f} | ε={eps:.3f} | FPS={fps:.1f}")

    plot_combined_metrics(episode_rewards, losses, fps_history, RESULTS_DIR, EXP_NUM)
    print(f"[INFO] Completed Config {EXP_NUM}")
    EXP_NUM += 1
