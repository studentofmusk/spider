import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time

class SpiderEnv(gym.Env):
    def __init__(self, render=False, enable_attack=False, train="stand", max_step=1000):
        super().__init__()

        self.train = train
        self.render = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.time_step = 1/240
        p.setTimeStep(self.time_step)
        self.max_step = max_step
        
        self.robot = None
        self.joint_indices = []

        self.NUM_JOINTS = 8
        self.JOINT_MIN = -1.57
        self.JOINT_MAX = 1.57

        self.min_lift = 0.045
        self.goal = np.array([1.0, 0.0]) # will be randomized in reset
        self.goal_threshold = 0.15       # success distance
        self.enable_attack = enable_attack

        if self.enable_attack:
            self.attack_interval = 50        # attack every 50 env step
            self.ball_lifetime = 120         # env steps (~5 seconds)


        # Action = delta joint angles
        self.action_space = spaces.Box(
            low=-0.25,
            high=0.25,
            shape=(self.NUM_JOINTS,),
            dtype=np.float32
        )

        # OBSERVATION = joint pos + joint vel + roll + pitch + goal_x + goal_y
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.NUM_JOINTS*2 + 2 + 2,),
            dtype=np.float32
        )

        self.reset()
    
    def reset(self, seed=None, options=None):

        self.total_reward = 0
        self.step_count = 0
        self.enable_balls = False
        self.balls = []

        # random goal in front area
        self.goal = np.array([
            np.random.uniform(0.8, 1.5),
            np.random.uniform(-0.6, 0.6)
        ])

        self.prev_goal_dist = None


        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        plane = p.loadURDF("plane.urdf")
        p.changeDynamics(plane, -1, lateralFriction=2.0)

        self.robot = p.loadURDF(
            "spider.urdf",
            [0, 0, 0.055],
            useFixedBase=False
        )

        self.joint_indices = []

        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)

        for j in self.joint_indices:
            p.changeDynamics(self.robot, j, lateralFriction=2.0)


        return self._get_obs(), {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        

        joint_pos = []
        joint_vel = []

        for s in joint_states:
            joint_pos.append(s[0])
            joint_vel.append(s[1])
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

        goal_vec = self.goal - np.array(base_pos[:2])

        # rotate goal into robot frame
        goal_local_x = np.cos(-yaw) * goal_vec[0] - np.sin(-yaw) * goal_vec[1]
        goal_local_y =  np.sin(-yaw) * goal_vec[0] + np.cos(-yaw) * goal_vec[1]

        obs = np.array(
            joint_pos + joint_vel + [roll, pitch] + [goal_local_x, goal_local_y],
            dtype=np.float32
        )

        return obs 

    def step(self, action):
        for i, joint_id in enumerate(self.joint_indices):
            current_pos = p.getJointState(self.robot, joint_id)[0]
            target_pos = np.clip(
                current_pos + action[i],
                self.JOINT_MIN,
                self.JOINT_MAX
            )
            target_vel = action[i] * 6.0

            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,
                # targetPosition=target_pos,
                targetVelocity=target_vel,
                force=20
            )
        
        for _ in range(4):
            p.stepSimulation()
            if self.render:
                time.sleep(self.time_step)
        
        self.step_count += 1
        if self.enable_balls and self.step_count % self.attack_interval == 0:
            self.ball_attack()

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        print(f"\rReward: {reward:.3f} | Total: {self.total_reward:.3f}", end="", flush=True)

        if self.enable_attack:
            if self.step_count > 20:
                self.enable_balls = True
            
            self.cleanup_balls()
        
        return obs, reward, done, False, {}

    # def _compute_reward(self):
    #     base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
    #     roll, pitch, _ = p.getEulerFromQuaternion(base_orn)

    #     stability_penalty = abs(roll) + abs(pitch)

    #     goal_dist = np.linalg.norm(self.goal-np.array(base_pos[:2]))

    #     if self.prev_goal_dist is None:
    #         self.prev_goal_dist = goal_dist

    #     progress = self.prev_goal_dist - goal_dist
    #     self.prev_goal_dist = goal_dist
        
        
    #     print("reward:", progress)

    #     if base_pos[2] == 0:
    #         return -5.0
        
        
    #     if base_pos[2] < self.min_lift:
    #         return -1 * ((self.min_lift - base_pos[2])/self.min_lift)
        
    #     balance_bonus = max(0.0, 1.0 - (abs(roll) + abs(pitch)))
    #     reward = 2.0 * progress
    #     reward = reward - 0.5 * stability_penalty 
    #     reward = reward + 0.3 * balance_bonus
        
    #     if self.enable_attack:
    #         impact_force = sum(
    #             c[9]
    #             for ball_id, _ in self.balls
    #             for c in p.getContactPoints(bodyA=self.robot, bodyB=ball_id)
    #         )
    #         reward -= 0.0005 * impact_force

    #     if goal_dist < self.goal_threshold:
    #         reward += 10.0

    #     return reward
    
    def _compute_reward(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

        lin_vel, ang_vel = p.getBaseVelocity(self.robot)

        # ---------- Goal direction ----------
        goal_vec = self.goal - np.array(base_pos[:2])
        goal_dist = np.linalg.norm(goal_vec)

        if goal_dist > 1e-6:
            goal_dir = goal_vec / goal_dist
        else:
            goal_dir = np.zeros(2)

        # velocity projected toward goal direction
        vel_toward_goal = lin_vel[0] * goal_dir[0] + lin_vel[1] * goal_dir[1]

        # ---------- Joint motion ----------
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        joint_vel = np.mean([abs(s[1]) for s in joint_states])

        # ---------- Reward components ----------
        reward = 0.0
        if self.train == "stand":
            # Balance (important but not dominant)
            stability_penalty = abs(roll) + abs(pitch)
            reward -= 0.5 * stability_penalty

            
            # 5️⃣ Height safety (no crawling flat)
            if base_pos[2] < self.min_lift:
                ratio = (self.min_lift-base_pos[2])/self.min_lift
                reward -= ratio * 1.0
            else:
                reward += 1.0

        else:
            # 1️⃣ Move forward (MAIN DRIVER)
            reward += 1.0 * vel_toward_goal

            # 2️⃣ Extra reward for raw forward velocity
            reward += 0.4 * np.clip(lin_vel[0], -1.0, 2.0)

            # 3️⃣ Balance (important but not dominant)
            stability_penalty = abs(roll) + abs(pitch)
            reward -= 0.3 * stability_penalty

            # 4️⃣ Anti-freeze penalty
            if joint_vel < 0.02:
                reward -= 0.3

            # 5️⃣ Height safety (no crawling flat)
            if base_pos[2] < self.min_lift:
                ratio = (self.min_lift-base_pos[2])/self.min_lift
                reward -= ratio * 1.0
            

            # 6️⃣ Time penalty (forces urgency)
            reward -= 0.01

            # 7️⃣ Goal reached bonus
            if goal_dist < self.goal_threshold:
                reward += 10.0

        return reward

    def _is_done(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)

        if abs(roll) > 0.5 or abs(pitch) > 0.5:
            return True
        
        if self.step_count > self.max_step:
            return True
        
        goal_dist = np.linalg.norm(self.goal-np.array(base_pos[:2]))
        
        if goal_dist < self.goal_threshold:
            return True
        # print(self.total_reward)
        return False
    
    def spawn_ball(self, position, velocity, radius=0.05, mass=2):
        col_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 1]
        )
        ball = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position
        )

        p.changeDynamics(
            ball,
            -1,
            lateralFriction=0.5,
            rollingFriction=0.01,
            spinningFriction=0.01,
            restitution=0.0
        )


        p.resetBaseVelocity(ball, linearVelocity=velocity)
        
        # ⬇️ store ball with spawn step
        self.balls.append((ball, self.step_count))
    
    def ball_attack(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        spawn_pos = [
            base_pos[0] + np.random.uniform(-0.8, 0.8),
            base_pos[1] + np.random.uniform(-0.8, 0.8),
            base_pos[2] + np.random.uniform(0.3, 0.5)
        ]

        target = np.array(base_pos)
        target[2] += 0.05   # aim slightly above COM
        direction = target - np.array(spawn_pos)
        direction = direction / np.linalg.norm(direction)

        speed = np.random.uniform(10.0, 15.0) # START VERY SAFE
        velocity = direction * speed

        self.spawn_ball(
            position=spawn_pos,
            velocity=velocity
        )

    def cleanup_balls(self):
        alive_balls = []

        for ball, spawn_step in self.balls:
            if self.step_count - spawn_step > self.ball_lifetime:
                p.removeBody(ball)
            else:
                alive_balls.append((ball, spawn_step))

        self.balls = alive_balls

