from stable_baselines3 import PPO
from spider_env import SpiderEnv
import os

TRAINED_MODEL_PATH = "spider_vel_walk.zip"
MODEL_PATH = "spider_vel_walk_with_ball.zip"

env = SpiderEnv(render=False, enable_attack=True, train="walk", max_step=2500)

if os.path.exists(TRAINED_MODEL_PATH):
    print("Loading existing model and continuing training...")
    model = PPO.load(
        TRAINED_MODEL_PATH,
        env=env,
        tensorboard_log="./logs/"
    )
else:
    print("No saved model found. Training from scratch...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./logs/"
    )

try:
    model.learn(total_timesteps=1_000_000)
except KeyboardInterrupt:
    print("Saving model...")
    model.save(MODEL_PATH)
    print("Saved")
