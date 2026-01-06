# Reinforcement-Learning

This project focuses on the comparative analysis of Reinforcement Learning (RL) algorithms within the Atari 2600 environment (via Gymnasium). Serving as a baseline for research, this repository implements and evaluates Value-Based and Policy-Based methods to solve classic discrete control tasks.

The project highlights the evolution from standard implementation to hyperparameter tuning, analyzing stability, convergence, and sample efficiency.

# 1. Envirnments:
1. Breakout
2. Pong
3. Space Invaders
4. Beam Rider
5. Freeway

# 2. Algorithms:
1. DQN (Deep Q-Network) - Value-Based
   
    The foundational deep RL algorithm that combines Q-Learning with deep neural networks. It utilizes Experience Replay and a Target Network to stabilize training by breaking correlations between consecutive samples.
   
2. PPO (Proximal Policy Optimization) - Policy-Based
   
   A policy gradient method that alternates between sampling data through interaction with the environment and optimizing a "surrogate" objective function using stochastic gradient descent. It balances exploration and exploitation by clipping policy updates to prevent destructive large steps.
   
3. C51(Categorical DQN) - Distributional Value-Based

   A distributional perspective on reinforcement learning. Instead of estimating only the expected mean Q-value (as DQN does), C51 models the entire probability distribution of returns using a categorical distribution with 51 fixed support points (atoms). By capturing the variance and uncertainty of future rewards, it provides richer training signals and significantly improves stability and asymptotic performance on Atari benchmarks.
  
# 3. Extensive hyperparameter optimization 
Learning rate, buffer size, and ent_coef (entropy coefficient) are tuned to analyze the trade-off between overfitting and exploration.

# 4. Metrics
Visualization of Reward per Episode, Loss, and Q-Values using Seaborn.
