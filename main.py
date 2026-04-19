import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("spider.urdf", [0, 0, 0.2], useFixedBase=False)

# Increase friction
p.changeDynamics(plane, -1, lateralFriction=2.0)
for j in range(p.getNumJoints(robot)):
    p.changeDynamics(robot, j, lateralFriction=2.0)

joint_ids = []
for i in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE:
        joint_ids.append(i)

print("Joints:", joint_ids)

while True:
    for i, j in enumerate(joint_ids):
        p.setJointMotorControl2(
            robot,
            j,
            p.VELOCITY_CONTROL,
            targetVelocity=5.0,
            force=40
        )
    p.stepSimulation()
    time.sleep(1/240)
