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
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward

GAMES_CONFIG = [
    #{"id": "PongNoFrameskip-v4", "v_min": -21.0, "v_max": 21.0, "clip": False, "frames": 5_000_000},
    {"id": "BreakoutNoFrameskip-v4", "v_min": -10.0, "v_max": 10.0, "clip": True, "frames": 5_000_000},
    #{"id": "FreewayNoFrameskip-v4", "v_min": 0.0, "v_max": 24.0, "clip": False, "frames": 5_000_000},
    #{"id": "SpaceInvadersNoFrameskip-v4", "v_min": -10.0, "v_max": 10.0, "clip": True, "frames": 5_000_000},
    #{"id": "BeamRiderNoFrameskip-v4", "v_min": -10.0, "v_max": 10.0, "clip": True, "frames": 5_000_000}
]

VARIATIONS = [
    {"name": "1_Baseline",       "params": {}}, 
    {"name": "2_Atoms_21",       "params": {"n_atoms": 21}},
    {"name": "3_Gamma_090",      "params": {"gamma": 0.90}},
    {"name": "4_Batch_128",      "params": {"batch_size": 128}},
    {"name": "5_LR_1e-4",        "params": {"lr": 1e-4}},
]

DEFAULTS = {
    "n_atoms": 51,
    "gamma": 0.99,
    "batch_size": 64,
    "lr": 2.5e-4,
    "buffer_size": 100_000,
    "min_replay": 10_000,
    "target_tau": 0.005,
    "train_every": 4,
}

RESULTS_DIR = os.path.join(os.getcwd(), "results_ablation_c51")
os.makedirs(RESULTS_DIR, exist_ok=True)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[SYSTEM] Accelerated Hardware: NVIDIA CUDA")
    torch.set_float32_matmul_precision("high")
    USE_AMP = True 
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[SYSTEM] Accelerated Hardware: Apple MPS (Metal)")
    USE_AMP = False 
else:
    DEVICE = torch.device("cpu")
    print("[SYSTEM] Hardware: CPU (Slow)")
    USE_AMP = False
    
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

class C51(nn.Module):
    def __init__(self, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, n_actions * n_atoms)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x).view(x.size(0), -1)
        logits = self.fc(x).view(-1, self.n_actions, self.n_atoms)
        return F.log_softmax(logits, dim=-1)

    def get_q_value(self, x):
        log_probs = self(x)
        return (log_probs.exp() * self.atoms).sum(dim=2)


def run_single_experiment(game_conf, var_conf):
    params = DEFAULTS.copy()
    params.update(var_conf["params"])
    
    N_ATOMS = params["n_atoms"]
    GAMMA = params["gamma"]
    BATCH_SIZE = params["batch_size"]
    LR = params["lr"]
    
    exp_id = f"{game_conf['id']}_{var_conf['name']}"
    print(f"\n[START] {exp_id} | Atoms:{N_ATOMS} | Gamma:{GAMMA} | Batch:{BATCH_SIZE} | LR:{LR}")

    env = gym.make(game_conf["id"])
    if game_conf["clip"]:
        env = TransformReward(env, lambda r: np.sign(r))
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=4, screen_size=84, terminal_on_life_loss=False)
    env = FrameStackObservation(env, 4)
    n_actions = env.action_space.n

    # networks
    policy_net = C51(n_actions, N_ATOMS, game_conf["v_min"], game_conf["v_max"]).to(DEVICE)
    target_net = C51(n_actions, N_ATOMS, game_conf["v_min"], game_conf["v_max"]).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR, eps=1e-5)
    replay = ReplayBuffer(params["buffer_size"])
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # variables
    state, _ = env.reset()
    episode_rewards = []
    fps_history = []
    episode_reward = 0
    frame_count = 0
    start_time = time.time()
    last_log_time = start_time
    
    delta_z = (game_conf["v_max"] - game_conf["v_min"]) / (N_ATOMS - 1)

    while frame_count < game_conf["frames"]:
        eps = max(0.05, 1.0 - (frame_count / 500_000) * 0.95)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
                action = policy_net.get_q_value(s_t).argmax(1).item()

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

        if len(replay) >= params["min_replay"] and frame_count % params["train_every"] == 0:
            s, a, r, s2, d = replay.sample(BATCH_SIZE)
            s = torch.as_tensor(s, device=DEVICE)
            a = torch.as_tensor(a, device=DEVICE).long()
            r = torch.as_tensor(r, device=DEVICE, dtype=torch.float32)
            s2 = torch.as_tensor(s2, device=DEVICE)
            d = torch.as_tensor(d, device=DEVICE, dtype=torch.float32)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    # double DQN Selection
                    next_actions = (target_net(s2).exp() * target_net.atoms).sum(2).argmax(1)
                    next_dist = target_net(s2).exp()[range(BATCH_SIZE), next_actions]

                    t_z = r.unsqueeze(1) + GAMMA * (1 - d.unsqueeze(1)) * target_net.atoms.unsqueeze(0)
                    t_z = t_z.clamp(min=game_conf["v_min"], max=game_conf["v_max"])
                    b = (t_z - game_conf["v_min"]) / delta_z
                    l, u = b.floor().long(), b.ceil().long()
                    l[(u > 0) * (l == u)] -= 1
                    
                    proj_dist = torch.zeros((BATCH_SIZE, N_ATOMS), device=DEVICE)
                    offset = torch.linspace(0, (BATCH_SIZE - 1) * N_ATOMS, BATCH_SIZE, device=DEVICE).long().unsqueeze(1)
                    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

                log_p = policy_net(s)[range(BATCH_SIZE), a]
                loss = - (proj_dist * log_p).sum(dim=1).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                tp.data.copy_(params["target_tau"] * pp.data + (1 - params["target_tau"]) * tp.data)

        if frame_count % 50_000 == 0:
            fps = 50_000 / (time.time() - last_log_time)
            last_log_time = time.time()
            fps_history.append(fps)
            avg = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            print(f"[{exp_id}] Step {frame_count//1000}k | Avg Reward: {avg:.2f} | FPS: {fps:.0f}")

    env.close()
    
    np.save(os.path.join(RESULTS_DIR, f"rewards_{exp_id}.npy"), episode_rewards)
    
    plt.figure()
    plt.plot(episode_rewards, alpha=0.3)
    if len(episode_rewards) > 50:
        plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), color='red')
    plt.title(f"Results: {exp_id}")
    plt.savefig(os.path.join(RESULTS_DIR, f"plot_{exp_id}.png"))
    plt.close()

    del policy_net, target_net, optimizer, replay
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"Plan: {len(GAMES_CONFIG)} games x {len(VARIATIONS)} variations = {len(GAMES_CONFIG)*len(VARIATIONS)} experiments.")
    
    for game in GAMES_CONFIG:
        for variation in VARIATIONS:
            try:
                run_single_experiment(game, variation)
            except Exception as e:
                print(f"!!! ERROR in {game['id']} - {variation['name']}: {e}")
