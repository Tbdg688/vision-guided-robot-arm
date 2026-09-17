import pybullet as p
import pybullet_data
import numpy as np
import time
from ultralytics import YOLO

# ===== 相机参数 =====
width, height = 640, 480
fov = 60
fx = width / (2 * np.tan(np.radians(fov / 2)))
fy = fx
cx, cy = width / 2, height / 2
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)

cam_pos = np.array([0.5, 0.0, 0.8])
cam_target = np.array([0.5, 0.0, 0.0])
cam_up = np.array([0, 1, 0])

forward = cam_target - cam_pos
forward = forward / np.linalg.norm(forward)
right = np.cross(forward, cam_up)
right = right / np.linalg.norm(right)
up = np.cross(right, forward)
R = np.array([right, -up, forward]).T
t = -R @ cam_pos


def pixel_to_3d(u, v, plane_z=0.05):
    pt_cam = np.linalg.inv(camera_matrix) @ np.array([u, v, 1.0])
    pt_cam = pt_cam / np.linalg.norm(pt_cam)
    direction_world = np.linalg.inv(R) @ pt_cam
    direction_world = direction_world / np.linalg.norm(direction_world)
    if abs(direction_world[2]) < 1e-6:
        return None
    t_scale = (plane_z - cam_pos[2]) / direction_world[2]
    point_3d = cam_pos + t_scale * direction_world
    return point_3d


def move_to(robot_id, end_effector_index, num_joints, target_pos, steps=800):
    """IK 求解 + 控制机械臂移动到目标位置"""
    joint_poses = p.calculateInverseKinematics(
        robot_id,
        end_effector_index,
        [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
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
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1./240.)


# ===== 连接仿真环境 =====
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
end_effector_index = 6
num_joints = p.getNumJoints(robot_id)

# 在场景里放几个方块（和生成数据集时一致）
import random
random.seed(42)
cube_ids = []
cube_positions = []
for i in range(3):
    x = random.uniform(0.35, 0.65)
    y = random.uniform(-0.15, 0.15)
    cube_id = p.loadURDF("cube_small.urdf", [x, y, 0.05])
    cube_ids.append(cube_id)
    cube_positions.append([x, y, 0.05])

# ===== 视觉检测 =====
print("===== 阶段 1: YOLO 视觉检测 =====")
model = YOLO(r"D:\Vision_Machine\runs\detect\cube_detector\weights\best.pt")
results = model(r"D:\Vision_Machine\dataset\images\0000.png")

box = results[0].boxes[0]
x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
u, v = (x1 + x2) / 2, (y1 + y2) / 2
target_pos = pixel_to_3d(u, v, plane_z=0.05)
print(f"目标 3D 坐标: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")

# 根据 3D 坐标，找到最近的真实方块
target_pos_np = np.array([target_pos[0], target_pos[1], target_pos[2]])
distances = [np.linalg.norm(target_pos_np - np.array(cp)) for cp in cube_positions]
closest_idx = int(np.argmin(distances))
cube_to_pick = cube_ids[closest_idx]
print(f"最近的方块索引: {closest_idx}, 距离: {distances[closest_idx]:.4f}")

# ===== 阶段 2: 移动到方块上方 =====
print("===== 阶段 2: 移动到方块上方 =====")
approach_pos = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2] + 0.1)]
print(f"接近点: {approach_pos}")
move_to(robot_id, end_effector_index, num_joints, approach_pos)
print("已到达方块上方")

# ===== 阶段 3: 末端下降到抓取位置 =====
print("===== 阶段 3: 下降到抓取位置 =====")
grasp_pos = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2] + 0.02)]
move_to(robot_id, end_effector_index, num_joints, grasp_pos)
print("已到达抓取位置")

# ===== 阶段 4: 创建约束，模拟夹爪闭合 =====
print("===== 阶段 4: 夹爪闭合 =====")
constraint_id = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=end_effector_index,
    childBodyUniqueId=cube_to_pick,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)
print(f"约束已创建, ID = {constraint_id}")

# ===== 阶段 5: 抬起 =====
print("===== 阶段 5: 抬起方块 =====")
lift_pos = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2] + 0.3)]
move_to(robot_id, end_effector_index, num_joints, lift_pos)

# 验证方块是否跟着抬起
cube_pos, _ = p.getBasePositionAndOrientation(cube_to_pick)
print(f"方块当前位置: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
print(f"抬起高度: {cube_pos[2]:.3f} 米（初始约 0.05）")

# ===== 阶段 6: 放置到新位置 =====
print("===== 阶段 6: 放置 =====")
place_pos = [0.4, 0.25, 0.2]
move_to(robot_id, end_effector_index, num_joints, place_pos)
p.removeConstraint(constraint_id)
print("约束已解除，方块放置完成")

# 再等几步，让方块稳定
for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

cube_pos_final, _ = p.getBasePositionAndOrientation(cube_to_pick)
print(f"方块最终位置: ({cube_pos_final[0]:.3f}, {cube_pos_final[1]:.3f}, {cube_pos_final[2]:.3f})")

time.sleep(3)
p.disconnect()
print("完成")