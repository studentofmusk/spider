# 🕷️ Spider RL: Physics-Based Reinforcement Learning Locomotion

## 📌 Overview

Spider RL is a reinforcement learning project where a multi-legged robot learns locomotion in a physics simulation.
The robot is trained using PPO to achieve **stable walking, balance, and goal-directed movement** without predefined gait patterns.

---

## 🚀 Key Features

* 🧠 Reinforcement Learning using PPO (Stable-Baselines3)
* 🌍 Custom Gym environment built with PyBullet
* 🕷️ 8-joint spider robot (continuous control)
* 🎯 Goal-based navigation with reward shaping
* ⚔️ Dynamic disturbance system (ball attack mode)
* ⚡ Real-time simulation with optimized control loop

---

## 🧠 System Architecture

### 1. Simulation Environment

* Built using PyBullet for realistic physics simulation ([PyBullet][1])
* Custom environment implemented using Gymnasium API
* Gravity, friction, and joint dynamics configured manually

👉 Environment class: `SpiderEnv` 

---

### 2. Observation Space

The agent observes:

* Joint positions & velocities (8 joints)
* Robot orientation (roll, pitch)
* Relative goal position (in robot frame)

---

### 3. Action Space

* Continuous control (8-dimensional)
* Each action controls joint velocity updates

---

### 4. Reward Design (Core Intelligence 🔥)

Custom reward function includes:

* ✅ Forward movement toward goal
* ✅ Stability (penalty for roll & pitch)
* ✅ Height constraint (avoid collapsing)
* ❌ Penalty for inactivity
* 🎯 Goal-reaching bonus

👉 Reward shaping implemented in `_compute_reward()` 

---

### 5. Training (PPO)

* Algorithm: Proximal Policy Optimization (PPO)
* Library: Stable-Baselines3
* Continuous action control for locomotion

👉 Training script: `train.py` 

---

### 6. Disturbance Learning (Advanced 🔥)

* Random **ball attacks** introduced during training
* Forces robot to learn **robust balance under disturbance**

👉 Implemented via:

* `ball_attack()`
* `spawn_ball()`
* `cleanup_balls()` 

---

## 📂 Project Structure

```bash
spider/
│── meshes/                # Robot mesh files
│── models/                # Trained PPO models (.zip)
│── logs/                  # Training logs (TensorBoard)
│── spider.urdf            # Robot structure definition
│── spider_env.py          # Custom RL environment
│── train.py               # Training script
│── test.py                # Run trained model
│── main.py                # Basic simulation test
```

---

## ▶️ Run Simulation (Basic)

```bash
python main.py
```

👉 Loads spider and runs velocity control loop 

---

## 🏋️ Train the Model

```bash
python train.py
```

* Trains PPO agent for locomotion
* Saves model as `.zip` file

---

## 🤖 Test Trained Model

```bash
python test.py
```

👉 Loads trained model and runs continuous control loop 

---

## 📊 Training Behavior

* Learns to maintain balance before movement
* Gradually develops coordinated leg motion
* Adapts to disturbances (ball attacks)
* Improves goal-directed navigation

---

## 🔮 Future Improvements

* Sim-to-real transfer (deploy on physical robot)
* Terrain randomization (uneven surfaces)
* Advanced RL (SAC / PPO + curriculum learning)
* Energy-efficient gait optimization

---

## 📬 Contact

**Farooq Khan**
📧 [farooqvrofficial@gmail.com](mailto:farooqvrofficial@gmail.com)
🔗 https://github.com/studentofmusk

---

[1]: https://pybullet.org/wordpress/index.php/2017/09/29/bullet-2-87-with-pybullet-robotics-reinforcement-learning-environments/?utm_source=chatgpt.com "Bullet 2.87 with pybullet robotics Reinforcement Learning ..."
