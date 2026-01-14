import gymnasium as gym
import ale_py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor


gym.register_envs(ale_py)


ENV_ID = "BreakoutNoFrameskip-v4"
TOTAL_STEPS = 5000000 
LOG_DIR = "./results/logs/ppo_breakout"
os.makedirs(LOG_DIR, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")


class ValueLogger(BaseCallback):
    def __init__(self, verbose=0):
       super().__init__(verbose)


    def _on_step(self) -> bool:
       if self.n_calls % 1000 == 0:
           try:
               obs = self.locals["new_obs"]
              
               if isinstance(obs, np.ndarray):
                   obs_tensor = torch.as_tensor(obs).to(self.model.device)
               else:
                   obs_tensor = obs
              
               with torch.no_grad():
                   values = self.model.policy.predict_values(obs_tensor)
              
               mean_val = values.mean().item()
               self.logger.record("custom/avg_value", mean_val)
              
           except Exception as e:
               pass
       return True


def plot_ppo_pretty(log_dir):
   print(f"--- Generating Plots from {log_dir} ---")
   monitor_path = os.path.join(log_dir, "monitor.csv")
   progress_path = os.path.join(log_dir, "progress.csv")

   if not os.path.exists(progress_path):
       print(f"[ERROR] No progress.csv found in {log_dir}. Wait for training to save logs.")
       return

   try:
       df_monitor = pd.read_csv(monitor_path, skiprows=1)
       df_progress = pd.read_csv(progress_path)
   except Exception as e:
       print(f"[ERROR] Could not read CSVs: {e}")
       return
  
   try:
       plt.style.use('seaborn-v0_8-darkgrid')
   except:
       plt.style.use('ggplot')

   fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
   fig.suptitle(f'PPO Training Metrics: {ENV_ID}', fontsize=16, weight='bold')

   def moving_average(data, window_size):
       if len(data) < window_size: return data
       return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

   if 'r' in df_monitor.columns:
       rewards = df_monitor['r'].values
       axs[0, 0].plot(rewards, alpha=0.15, color='gray', label='Raw Episode Reward')
       window = 50
       if len(rewards) >= window:
           avg_rewards = moving_average(rewards, window)
           x_range = np.arange(len(rewards) - len(avg_rewards), len(rewards))
           axs[0, 0].plot(x_range, avg_rewards, color='#1f77b4', linewidth=2, label=f'Avg Reward ({window})')
      
       axs[0, 0].set_title('Episode Rewards', weight='bold')
       axs[0, 0].set_ylabel('Reward')
       axs[0, 0].set_xlabel('Episodes')
       axs[0, 0].legend(loc='upper left')


   loss_key = 'train/value_loss'
   if loss_key in df_progress.columns:
       subset = df_progress.dropna(subset=[loss_key])
       steps = subset['time/total_timesteps']
       losses = subset[loss_key].values
      
       loss_window = max(1, int(len(losses) * 0.05))
       avg_losses = moving_average(losses, loss_window)
       x_loss = steps[-len(avg_losses):]
      
       axs[0, 1].plot(x_loss, avg_losses, color='#d62728', linewidth=2)
       axs[0, 1].set_title(f'Critic Value Loss (Smoothed)', weight='bold')
       axs[0, 1].set_ylabel('Loss')
       axs[0, 1].set_xlabel('Total Steps')
   else:
       axs[0, 1].text(0.5, 0.5, 'No Loss Data Found', ha='center')

   val_key = 'custom/avg_value'
   if val_key in df_progress.columns:
       subset = df_progress.dropna(subset=[val_key])
      
       if len(subset) > 0:
           steps = subset['time/total_timesteps']
           values = subset[val_key].values
          
           val_window = max(1, int(len(values) * 0.05))
           avg_values = moving_average(values, val_window)
           x_val = steps[-len(avg_values):]


           axs[1, 0].plot(x_val, avg_values, color='#2ca02c', linewidth=2)
           axs[1, 0].set_title('Mean State Value $V(s)$ (Critic)', weight='bold')
           axs[1, 0].set_ylabel('Estimated Value')
           axs[1, 0].set_xlabel('Total Steps')
       else:
           axs[1, 0].text(0.5, 0.5, 'Data present but all NaNs', ha='center')
   else:
       axs[1, 0].text(0.5, 0.5, 'No Value Data (Callback Issue)', ha='center')

   fps_key = 'time/fps'
   if fps_key in df_progress.columns:
       subset = df_progress.dropna(subset=[fps_key])
       axs[1, 1].plot(subset['time/total_timesteps'], subset[fps_key], color='#9467bd', linewidth=2)
       axs[1, 1].set_title('Training Speed (FPS)', weight='bold')
       axs[1, 1].set_ylabel('FPS')
       axs[1, 1].set_xlabel('Total Steps')


   plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  
   save_path = os.path.join(log_dir, "ppo_training_summary_fixed.png")
   plt.savefig(save_path)
   print(f"[INFO] Pretty plots saved to: {save_path}")
   plt.close()

if __name__ == "__main__":
   print(f"--- Setting up Atari Environment: {ENV_ID} ---")
   env = make_atari_env(ENV_ID, n_envs=8, seed=42)
   env = VecFrameStack(env, n_stack=4)
   env = VecMonitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))

   print("--- Initializing PPO Agent ---")
   model = PPO(
       "CnnPolicy",
       env,
       verbose=1,
       device=device,
       learning_rate=2.5e-4,     
       n_steps=128,      
       batch_size=256, 
       n_epochs=4,  
       clip_range=0.1,
       ent_coef=0.01, 
       vf_coef=0.5, 
   )

   new_logger = configure(LOG_DIR, ["stdout", "csv"])
   model.set_logger(new_logger)

   print(f"--- Training Started for {TOTAL_STEPS} steps ---")
   model.learn(total_timesteps=TOTAL_STEPS, callback=ValueLogger())
   print("--- Training Finished ---")

   save_path = os.path.join(LOG_DIR, "ppo_breakout_model")
   model.save(save_path)
   print(f"[INFO] Model saved to {save_path}.zip")

   plot_ppo_pretty(LOG_DIR)
