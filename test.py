from stable_baselines3 import PPO
from spider_env import SpiderEnv

env = SpiderEnv(render=True, enable_attack=False, train="walk", max_step=3000)
model = PPO.load("spider_vel_walk_with_ball")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
