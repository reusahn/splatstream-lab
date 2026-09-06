import numpy as np
from splatstream.synthetic import make_scene
from splatstream.camera import default_cameras
from splatstream.cpu_renderer import render
from splatstream.metrics import psnr, ssim

def test_renderer_shape_and_range():
    s = make_scene(80, seed=5)
    img = render(s, default_cameras()[0], 64, 64)
    assert img.shape == (64,64,3)
    assert img.min() >= 0
    assert img.max() <= 1

def test_identity_quality():
    s = make_scene(80, seed=6)
    a = render(s, default_cameras()[0], 64, 64)
    b = render(s.copy(), default_cameras()[0], 64, 64)
    assert psnr(a,b) == float("inf")
    assert ssim(a,b) > 0.9999
