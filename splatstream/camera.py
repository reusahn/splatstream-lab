from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_deg: float = 52.0

def camera_basis(camera: Camera):
    eye = np.asarray(camera.eye, dtype=np.float32)
    target = np.asarray(camera.target, dtype=np.float32)
    up = np.asarray(camera.up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up) + 1e-12
    return eye, right, true_up, forward

def default_cameras():
    return [
        Camera(np.array([0.0,0.1,0.0]), np.array([0.0,0.0,4.0]), np.array([0,1,0])),
        Camera(np.array([1.8,0.35,0.45]), np.array([0.0,0.0,4.0]), np.array([0,1,0])),
        Camera(np.array([-1.8,0.25,0.7]), np.array([0.0,0.0,4.1]), np.array([0,1,0])),
        Camera(np.array([0.2,1.15,0.7]), np.array([0.0,0.0,4.0]), np.array([0,1,0])),
    ]
