import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

end_effector_index = 6
num_joints = p.getNumJoints(robot_id)

target_position = [0.5, 0.0, 0.3]

# 只指定位置，不加姿态约束，但带上关节限制和迭代参数
joint_poses = p.calculateInverseKinematics(
    robot_id,
    end_effector_index,
    target_position,
    lowerLimits=[-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
    upperLimits=[2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054],
    jointRanges=[5.934, 4.188, 5.934, 4.188, 5.934, 4.188, 6.108],
    restPoses=[0, 0, 0, 0, 0, 0, 0],
    maxNumIterations=200,
    residualThreshold=0.0001
)

for i in range(num_joints):
    p.setJointMotorControl2(
        bodyIndex=robot_id,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=joint_poses[i],
        force=500
    )

for _ in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

link_state = p.getLinkState(robot_id, end_effector_index)
actual_position = link_state[0]
print(f"目标位置: {target_position}")
print(f"实际位置: {actual_position}")

error = np.linalg.norm(np.array(target_position) - np.array(actual_position))
print(f"误差: {error:.4f} 米")

time.sleep(3)
p.disconnect()