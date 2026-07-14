# DQN/DDQN for Lunar Lander

## 📖 Overview
This repository implements classic **Deep Q‑Network (DQN)** and its variants (Double‑DQN, Dueling‑DQN, etc.) to solve the **Lunar Lander** environment from the `gymnasium` (formerly OpenAI Gym) suite. The agent learns to control a lunar lander spacecraft, safely touching down on a target pad while minimising fuel usage.

The notebook `Lunar_Lander_Final.ipynb` walks through:
- Problem description and state/action spaces
- Reward structure and termination conditions
- Theory behind Q‑learning and DQN
- Network architecture and replay buffer design
- Training loop with epsilon‑greedy exploration
- Experimental results with different hyper‑parameters

## 🛠️ Environment & Dependencies
The code targets **Python 3.11** and runs on Windows (or Linux/macOS with minor adjustments).

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Core libraries
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install gymnasium[box2d] numpy tqdm matplotlib imageio
```
> **Note**: The original notebook used `apt‑get install swig` for the `box2d-py` dependency. On Windows you can install the pre‑compiled wheel via `pip install box2d-py` (handled automatically by `gymnasium[box2d]`).

## 🚀 Quick Start
1. **Open the notebook** – `Lunar_Lander_Final.ipynb` and run the cells sequentially.
2. **Or run the training script directly** (the core logic lives in the notebook and can be extracted to a `.py` file):
   ```bash
   python -c "import gymnasium as gym; from Lunar_Lander_Final import trainAgent_default;\
   env = gym.make('LunarLander-v3');\
   policy_net, scores = trainAgent_default(env, learning_rate=0.001, gamma=0.99)"
   ```
3. Observe the printed average reward per 100‑episode block and the final policy network.

## 📚 Algorithm Details
### 1️⃣ Q‑Learning Recap
The classic update rule:
$$
Q(s, a) \leftarrow Q(s, a) + \alpha\big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big]
$$
where \(\alpha\) is the learning rate and \(\gamma\) the discount factor.

### 2️⃣ Deep Q‑Network (DQN)
- **Network** – three fully‑connected layers (`fc1`, `fc2`, `fc3`) with ReLU activations.
- **Replay Buffer** – stores tuples \((s, a, r, s', done)\) and samples random minibatches to break correlation.
- **Target Network** – a separate network whose parameters are periodically copied from the policy network to stabilise learning.

### 3️⃣ Epsilon‑Greedy Policy
Select a random action with probability \(\varepsilon\) (exploration) or the greedy action otherwise (exploitation):
$$
 a = \begin{cases}\text{random} & \text{with prob. } \varepsilon \\
 \arg\max_a Q(s, a) & \text{with prob. } 1-\varepsilon\end{cases}
$$
\(\varepsilon\) decays each episode (e.g., start = 1.0, end = 0.01, decay = 0.995).

## 🏋️ Training Loop (simplified)
```python
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = select_action(state, eps, policy_net, action_size)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push(state, action, reward, next_state, done)
        state = next_state
        if len(memory) > batch_size:
            # Sample, compute loss, back‑propagate
            ...
        if total_steps % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    eps = max(eps_end, eps * eps_decay)
```

## 📊 Results (from the notebook)
| Episode Block | Avg Reward | ε (epsilon) |
|---------------|------------|-------------|
| 1‑100   | **-138.85** | 0.606 |
| 101‑200 | **-40.59**  | 0.367 |
| 201‑300 | **14.39**   | 0.222 |
| 301‑400 | **-45.69**  | 0.135 |
| 401‑500 | **-71.55**  | 0.082 |
| 501‑600 | **-71.83**  | 0.049 |
| 601‑700 | **-79.58**  | 0.030 |
| 701‑800 | **-78.89**  | 0.018 |
| 801‑900 | **-84.93**  | 0.011 |
| 901‑1000| **-83.43**  | 0.010 |

The agent quickly learns a useful policy (positive reward around episode 300) but later performance fluctuates, highlighting the sensitivity to hyper‑parameters such as learning rate, discount factor, and replay buffer size.

## ⚙️ Hyper‑Parameters Used
```python
n_episodes = 1000
batch_size = 64
gamma = 0.99          # discount factor
learning_rate = 0.001
memory_size = 100_000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
target_update = 1000  # steps
```
You can experiment with **Double‑DQN**, **Dueling‑DQN**, or **Rainbow DQN** by modifying the network architecture and loss calculation as described in the notebook.

## 📂 Project Structure
```
DQN-DDQN-for-lunar-lander/
│   Lunar_Lander_Final.ipynb   # Full walkthrough (markdown + code)
│   README.md                 # THIS FILE
│   requirements.txt          # (optional) pip freeze list
└───utils/ (if you extract helper functions later)
```

## 📝 License
This example code is provided for educational purposes under the **MIT License**. Feel free to adapt and extend it for research or personal projects.

---
*Happy landing!*
