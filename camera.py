import os
import ctypes
import cv2
import numpy as np

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

def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        raise RuntimeError("No USB camera found.")

    stDevInfo = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    
    cam = MvCamera()
    FPS = 120.0 #ENTER Hz HERE 
    cam_config(cam, stDevInfo, FPS)

    payload = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", payload)
    payload_size = int(payload.nCurValue)

    data_buf = (ctypes.c_ubyte * payload_size)()

    fps = 0
    frame_count = 0
    prev_time = cv2.getTickCount()

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, stFrameInfo, 1000)
        if ret != 0:
            continue

        w, h = stFrameInfo.nWidth, stFrameInfo.nHeight

        img_rgb = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        frame_count += 1
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - prev_time) / cv2.getTickFrequency()
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            prev_time = current_time

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(img_bgr, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("HIK ROBOT Live (RGB)", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()