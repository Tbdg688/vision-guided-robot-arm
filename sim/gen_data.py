import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
import random

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 相机参数
width, height = 640, 480
fov = 60
fx = width / (2 * np.tan(np.radians(fov / 2)))
fy = fx
cx, cy = width / 2, height / 2
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)

# 相机在 3D 空间中的位置和朝向（俯视桌面）
cam_pos = np.array([0.5, 0.0, 0.8])
cam_target = np.array([0.5, 0.0, 0.0])
cam_up = np.array([0, 1, 0])   # 关键改动：不能与视线方向平行

# 计算相机外参矩阵
forward = cam_target - cam_pos
forward = forward / np.linalg.norm(forward)
right = np.cross(forward, cam_up)
right = right / np.linalg.norm(right)
up = np.cross(right, forward)
R = np.array([right, -up, forward]).T
t = -R @ cam_pos

def project_to_2d(point_3d):
    """将 3D 世界坐标投影到 2D 像素坐标，带 NaN 保护"""
    pt_cam = R @ np.array(point_3d) + t
    if pt_cam[2] <= 0.001:
        return None
    pt_2d = camera_matrix @ pt_cam
    x = pt_2d[0] / pt_2d[2]
    y = pt_2d[1] / pt_2d[2]
    if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
        return None
    return (int(x), int(y))

for idx in range(50):
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    cubes = []
    for i in range(3):
        x = random.uniform(0.35, 0.65)
        y = random.uniform(-0.15, 0.15)
        cube_id = p.loadURDF("cube_small.urdf", [x, y, 0.05])
        cubes.append(cube_id)

    for _ in range(30):
        p.stepSimulation()

    # 创建空白画布（浅灰色背景）
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 200

    # 画桌面（地面平面）
    table_corners = [[0.2, -0.4, 0.0], [0.8, -0.4, 0.0],
                     [0.8, 0.4, 0.0], [0.2, 0.4, 0.0]]
    table_2d = [project_to_2d(c) for c in table_corners]
    if all(pt is not None for pt in table_2d):
        pts = np.array(table_2d, dtype=np.int32)
        cv2.fillPoly(canvas, [pts], (180, 180, 180))

    labels = []
    for cube_id in cubes:
        pos, orn = p.getBasePositionAndOrientation(cube_id)
        half = 0.025  # cube_small 边长约 0.05
        corners_3d = []
        for dx in [-half, half]:
            for dy in [-half, half]:
                for dz in [-half, half]:
                    corners_3d.append([pos[0]+dx, pos[1]+dy, pos[2]+dz])

        corners_2d = [project_to_2d(c) for c in corners_3d]
        corners_2d = [c for c in corners_2d if c is not None]
        if len(corners_2d) < 4:
            continue

        xs = [c[0] for c in corners_2d]
        ys = [c[1] for c in corners_2d]
        x_min, x_max = max(0, min(xs)), min(width, max(xs))
        y_min, y_max = max(0, min(ys)), min(height, max(ys))

        if x_max <= x_min or y_max <= y_min:
            continue

        # 画方块（蓝色填充 + 深色边框）
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (200, 120, 50), -1)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (100, 60, 20), 2)

        # YOLO 标注
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    cv2.imwrite(f"dataset/images/{idx:04d}.png", canvas)
    with open(f"dataset/labels/{idx:04d}.txt", "w") as f:
        f.write("\n".join(labels))

p.disconnect()
print("dataset generated")