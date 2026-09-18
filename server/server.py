from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import io
from PIL import Image
import pybullet as p
import pybullet_data
import time
import random

app = FastAPI()
model = YOLO(r"D:\Vision_Machine\weights\best.pt")

# ===== 相机参数 =====
width, height = 640, 480
fov = 60
fx = width / (2 * np.tan(np.radians(fov / 2)))
fy = fx
cx, cy = width / 2, height / 2
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

cam_pos = np.array([0.5, 0.0, 0.8])
cam_target = np.array([0.5, 0.0, 0.0])
cam_up = np.array([0, 1, 0])
forward = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
right = np.cross(forward, cam_up) / np.linalg.norm(np.cross(forward, cam_up))
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
    return cam_pos + t_scale * direction_world


# ===== PyBullet 懒加载 =====
_pybullet_initialized = False
robot_id = None
end_effector_index = 6
num_joints = 7
cube_ids = []
cube_positions = []


def init_pybullet():
    global _pybullet_initialized, robot_id, cube_ids, cube_positions
    if _pybullet_initialized:
        return
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    random.seed(42)
    cube_ids = []
    cube_positions = []
    for i in range(3):
        x = random.uniform(0.35, 0.65)
        y = random.uniform(-0.15, 0.15)
        cube_id = p.loadURDF("cube_small.urdf", [x, y, 0.05])
        cube_ids.append(cube_id)
        cube_positions.append([x, y, 0.05])

    _pybullet_initialized = True


def move_to(target_pos, steps=800):
    joint_poses = p.calculateInverseKinematics(
        robot_id, end_effector_index,
        [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
        lowerLimits=[-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054],
        upperLimits=[2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054],
        jointRanges=[5.934, 4.188, 5.934, 4.188, 5.934, 4.188, 6.108],
        restPoses=[0, 0, 0, 0, 0, 0, 0],
        maxNumIterations=200,
        residualThreshold=0.0001
    )
    for i in range(num_joints):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                                targetPosition=joint_poses[i], force=500)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1./240.)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model(image)
    boxes_data = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        u, v = (x1 + x2) / 2, (y1 + y2) / 2

        pos_3d = pixel_to_3d(u, v, plane_z=0.05)
        if pos_3d is not None:
            boxes_data.append({
                "class_id": int(box.cls[0].cpu().numpy()),
                "confidence": round(conf, 3),
                "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
                "center_2d": [float(u), float(v)],
                "position_3d": [round(float(pos_3d[0]), 4),
                                round(float(pos_3d[1]), 4),
                                round(float(pos_3d[2]), 4)]
            })

    return JSONResponse({"count": len(boxes_data), "objects": boxes_data})


@app.post("/pick")
async def pick(file: UploadFile = File(...)):
    init_pybullet()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model(image)
    if len(results[0].boxes) == 0:
        return JSONResponse({"success": False, "message": "未检测到目标"})

    box = results[0].boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    target_pos = pixel_to_3d(u, v)
    if target_pos is None:
        return JSONResponse({"success": False, "message": "坐标反投影失败"})

    # 匹配最近的真实方块
    target_np = np.array(target_pos)
    distances = [np.linalg.norm(target_np - np.array(cp)) for cp in cube_positions]
    closest_idx = int(np.argmin(distances))
    cube_to_pick = cube_ids[closest_idx]

    # 执行抓取
    move_to([target_pos[0], target_pos[1], target_pos[2] + 0.1])
    move_to([target_pos[0], target_pos[1], target_pos[2] + 0.02])

    constraint_id = p.createConstraint(
        robot_id, end_effector_index, cube_to_pick, -1,
        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0]
    )

    move_to([target_pos[0], target_pos[1], target_pos[2] + 0.3])

    cube_pos, _ = p.getBasePositionAndOrientation(cube_to_pick)
    lift_height = float(cube_pos[2])

    move_to([0.4, 0.25, 0.2])
    p.removeConstraint(constraint_id)

    return JSONResponse({
        "success": True,
        "target_3d": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])],
        "lift_height": lift_height,
        "message": f"抓取成功，方块从 0.05 米抬升到 {lift_height:.3f} 米"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)