"""
Live camera feed with the same best blue filter as filters.py.
Uses HIKRobot MVS camera; applies gamma 1.2 then best filter (tight HSV + env suppression).
Only blue lights visible; ignores background blues and whites (uses image_env.png).
Press 'q' to quit.
"""

import os
import time
import ctypes
import cv2
import numpy as np
from pathlib import Path

# Same DLL setup as camera.py
DLL_DIRS = [
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
]

dll_dir = next((d for d in DLL_DIRS if os.path.exists(os.path.join(d, "MvCameraControl.dll"))), None)
if dll_dir is None:
    raise FileNotFoundError("MvCameraControl.dll not found. Install HIKRobot MVS (Windows x64).")

os.add_dll_directory(dll_dir)
os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

from MVS.MvCameraControl_class import *


# Exact same as filters.py best filter
GAMMA = 1.2
BEST_HUE_MIN, BEST_HUE_MAX = 85, 118
BEST_SAT_MIN, BEST_VAL_MIN = 90, 190
ENV_V_DELTA = 18
VAL_ABSOLUTE = 248
IMAGE_ENV_NAME = "image_env.png"


def cam_config(cam: MvCamera, stDevInfo, FPS: float):
    cam.MV_CC_CreateHandle(stDevInfo)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
    cam.MV_CC_SetEnumValue("ExposureAuto", 2)
    cam.MV_CC_SetEnumValue("GainAuto", 2)
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 2)
    cam.MV_CC_StartGrabbing()
    cam.MV_CC_SetEnumValue("TriggerMode", 0)
    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
    cam.MV_CC_SetFloatValue("AcquisitionFrameRate", FPS)


def load_env():
    """Load image_env.png for background suppression (same as filters.py)."""
    base = Path(__file__).resolve().parent
    path = base / IMAGE_ENV_NAME
    if not path.exists():
        return None
    return cv2.imread(str(path))


def apply_gamma(img_bgr, gamma=GAMMA):
    """Gamma correction — same as filters.py."""
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def apply_blue_mask_best(img_bgr, img_env_bgr=None):
    """
    Best filter: tight HSV + require brighter than env (exact same as filters.py).
    Rejects matte blue, whites/grays, faint blues; only emissive blue lights pass.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_current = hsv[:, :, 2]
    lower = np.array([BEST_HUE_MIN, BEST_SAT_MIN, BEST_VAL_MIN])
    upper = np.array([BEST_HUE_MAX, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    if img_env_bgr is not None:
        if img_env_bgr.shape[:2] != img_bgr.shape[:2]:
            img_env_bgr = cv2.resize(img_env_bgr, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        hsv_env = cv2.cvtColor(img_env_bgr, cv2.COLOR_BGR2HSV)
        v_env = hsv_env[:, :, 2].astype(np.int32)
        v_cur = v_current.astype(np.int32)
        brighter_than_env = (v_cur > v_env + ENV_V_DELTA) | (v_current >= VAL_ABSOLUTE)
        mask = cv2.bitwise_and(mask, np.where(brighter_than_env, 255, 0).astype(np.uint8))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = np.zeros_like(img_bgr)
    out[mask > 0] = img_bgr[mask > 0]
    return out


def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        raise RuntimeError("No USB camera found.")

    stDevInfo = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    cam = MvCamera()
    FPS = 60.0
    cam_config(cam, stDevInfo, FPS)

    payload = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", payload)
    payload_size = int(payload.nCurValue)
    data_buf = (ctypes.c_ubyte * payload_size)()

    img_env = load_env()
    if img_env is None:
        print("image_env.png not found; using best filter without env suppression.")

    t_prev = time.perf_counter()
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, stFrameInfo, 1000)
        if ret != 0:
            continue

        w, h = stFrameInfo.nWidth, stFrameInfo.nHeight
        img_rgb = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Exact same pipeline as filters.py: gamma then best filter (env + tight HSV)
        gamma_img = apply_gamma(img_bgr)
        out = apply_blue_mask_best(gamma_img, img_env)

        t_now = time.perf_counter()
        fps = 1.0 / (t_now - t_prev) if t_now > t_prev else 0.0
        t_prev = t_now
        cv2.putText(out, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Blue lights (best filter + env) — Live", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
