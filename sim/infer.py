import pybullet as p
import pybullet_data
import numpy as np
import cv2
from ultralytics import YOLO

# ===== 相机参数（与生成数据集时保持一致）=====
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
    """像素坐标反投影到 z=plane_z 平面上的 3D 世界坐标"""
    # 像素坐标 → 相机坐标系下的方向向量
    pt_cam = np.linalg.inv(camera_matrix) @ np.array([u, v, 1.0])
    pt_cam = pt_cam / np.linalg.norm(pt_cam)

    # 相机坐标系方向 → 世界坐标系方向
    direction_world = np.linalg.inv(R) @ pt_cam
    direction_world = direction_world / np.linalg.norm(direction_world)

    # 射线与平面 z = plane_z 求交
    if abs(direction_world[2]) < 1e-6:
        return None
    t_scale = (plane_z - cam_pos[2]) / direction_world[2]
    point_3d = cam_pos + t_scale * direction_world
    return point_3d


# ===== 加载模型并推理 =====
model = YOLO(r"D:\Vision_Machine\runs\detect\cube_detector\weights\best.pt")
results = model(r"D:\Vision_Machine\dataset\images\0000.png")

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    pos_3d = pixel_to_3d(u, v, plane_z=0.05)
    if pos_3d is not None:
        print(f"检测框中心: ({u:.0f}, {v:.0f}) → 3D 坐标: "
              f"({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})")