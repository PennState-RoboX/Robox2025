import os
import ctypes
import cv2
import numpy as np


import os

DLL_DIRS = [
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files (x86)\MVS\Applications\Win64",
]

dll_dir = next((d for d in DLL_DIRS if os.path.exists(os.path.join(d, "MvCameraControl.dll"))), None)
if dll_dir is None:
    raise FileNotFoundError("MvCameraControl.dll not found. Install HIKRobot MVS (Windows x64).")

os.add_dll_directory(dll_dir)
os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

from MVS.MvCameraControl_class import *

def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)

    stDevInfo = ctypes.cast(
        deviceList.pDeviceInfo[0],
        ctypes.POINTER(MV_CC_DEVICE_INFO)
    ).contents

    cam = MvCamera()
    cam.MV_CC_CreateHandle(stDevInfo)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_StartGrabbing()

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    buf_size = 50 * 1024 * 1024
    data_buf = (ctypes.c_ubyte * buf_size)()

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, buf_size, stFrameInfo, 1000)
        if ret != 0:
            continue

        w, h = stFrameInfo.nWidth, stFrameInfo.nHeight
        frame = np.frombuffer(data_buf, dtype=np.uint8, count=w * h)
        img = frame.reshape(h, w)

        cv2.imshow("HIK ROBOT Live", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
