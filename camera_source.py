import json
import logging
import ctypes
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from camera_params import camera_params, DepthSource
from hik_driver import *
from MVS.MvCameraControl_class import *
from camera_blue_filter import apply_gamma, apply_blue_mask_best, load_env

logger = logging.getLogger(__name__)

RS_DEPTH_CAPTURE_RES = (640, 480)

# HIK camera wrapper adapted from camera_blue_filter.py
class HikCameraSource:
    def __init__(self, active_cam_config: Dict, use_blue_filter: bool = False, fps: float = 60.0):
        self.active_cam_config = active_cam_config
        self.use_blue_filter = use_blue_filter
        self.img_env = load_env() if use_blue_filter else None

        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            raise RuntimeError("No HIK camera found.")

        stDevInfo = ctypes.cast(device_list.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

        self.cam = MvCamera()
        self.cam.MV_CC_CreateHandle(stDevInfo)
        self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        self.cam.MV_CC_SetFloatValue("ExposureTime", self.active_cam_config.get('exposure', {}).get('blue', 5000.0))
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        self.cam.MV_CC_SetFloatValue("Gain", 8.0)
        self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 1)
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", fps)
        self.cam.MV_CC_StartGrabbing()

        payload = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("PayloadSize", payload)
        self.payload_size = int(payload.nCurValue)
        self.data_buf = (ctypes.c_ubyte * self.payload_size)()
        self.frame_info = MV_FRAME_OUT_INFO_EX()

    def get_frames(self):
        ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.payload_size, self.frame_info, 1000)
        if ret != 0:
            return None, None

        w = self.frame_info.nWidth
        h = self.frame_info.nHeight

        img_rgb = np.frombuffer(self.data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.use_blue_filter:
            gamma_img = apply_gamma(img_bgr)
            filtered = apply_blue_mask_best(gamma_img, self.img_env)
            return filtered, None

        return img_bgr, None

    def release(self):
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()


# Unified image acquisition class for different types of cameras
class CameraSource:
    def __init__(self, default_config: Dict, target_color: str, cv_device_index: int = 0,
                 recording_source: Optional[Path] = None, recording_dest: Optional[Path] = None):
        assert recording_source is None or recording_dest is None
        self.recording_source=recording_source
        self._rs_pipeline = None
        self._rs_frame_aligner = None
        self._cv_color_cap = None
        self._cv_depth_cap = None
        self.hik_frame_cap = None
        self.color_frame_writer = None
        self.depth_frame_writer = None
        self.active_cam_config = None
        self.hik_frame_init=None

        if recording_source is None:
            self.active_cam_config = default_config

            # RealSense code commented out - using HIK camera instead
            # try:
            #     import pyrealsense2 as rs
            #     # Configure depth and color streams
            #     pipeline = rs.pipeline()
            #     config = rs.config()
            #
            #
            #     # Get device product line for setting a supporting resolution
            #     pipeline_wrapper = rs.pipeline_wrapper(pipeline)
            #     pipeline_profile = config.resolve(pipeline_wrapper)
            #     device = pipeline_profile.get_device()
            #     device_name = str(device.get_info(rs.camera_info.name))
            #
            #     if device_name in camera_params:
            #         self.active_cam_config = camera_params[device_name]
            #     else:
            #         logger.warning(
            #             f'Unknown device name: "{device_name}". Falling back to default configuration.')
            #
            #     config.enable_stream(rs.stream.color, self.active_cam_config['capture_res'][0],
            #                          self.active_cam_config['capture_res'][1], rs.format.bgr8,
            #                          self.active_cam_config['frame_rate'])
            #
            #     if self.active_cam_config['depth_source'] == DepthSource.STEREO:
            #         config.enable_stream(rs.stream.depth, RS_DEPTH_CAPTURE_RES[0], RS_DEPTH_CAPTURE_RES[1],
            #                              rs.format.z16,
            #                              self.active_cam_config['frame_rate'])
            #         frame_aligner = rs.align(rs.stream.color)
            #     else:
            #         frame_aligner = None
            #
            #     # Start streaming
            #     pipeline.start(config)
            #
            #     # Get the sensor once at the beginning. (Sensor index: 1)
            #     sensor = pipeline.get_active_profile(
            #     ).get_device().query_sensors()[1]
            #
            #     # Set the exposure anytime during the operation
            #     sensor.set_option(
            #         rs.option.exposure, self.active_cam_config['exposure'][target_color])
            #
            #     self._rs_pipeline = pipeline
            #     self._rs_frame_aligner = frame_aligner
            # except ImportError:
            #     logger.warning(
            #         'Intel RealSense backend is not available; pyrealsense2 could not be imported')
            # except RuntimeError as ex:
            #     if len(ex.args) >= 1 and 'No device connected' in ex.args[0]:
            #         logger.warning('No RealSense device was found')
            #     else:
            #         raise
            # if realsense is not available
            if self._rs_pipeline is None:
                # cap = cv2.VideoCapture()
                #
                # # the number here depends on your device's camera, usually default with 0
                # if cap.isOpened():
                #
                #     cap.open(cv_device_index)
                #
                #     cap.set(cv2.CAP_PROP_FOURCC,
                #             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                #     # os.system('v4l2-ctl --device=/dev/video1 --set-ctrl=exposure_auto=2')
                #     cap.set(cv2.CAP_PROP_EXPOSURE,
                #             self.active_cam_config['exposure'][target_color])
                #     cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                #             self.active_cam_config['capture_res'][0])
                #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                #             self.active_cam_config['capture_res'][1])
                #     cap.set(cv2.CAP_PROP_FPS, self.active_cam_config['frame_rate'])
                #     self._cv_color_cap = cap
                # if built-in camera not available, open hik
                # else:
                self.hik_frame_init = hik_init()


        else:
            cam_config_path = recording_source.with_name(
                recording_source.name + '.config.json')
            with open(cam_config_path, 'r', encoding='utf8') as cam_config_file:
                self.active_cam_config = json.load(cam_config_file)

            # color_frame_path = recording_source.with_name(
            #     recording_source.name + '.color.mp4')
            # depth_frame_path = recording_source.with_name(
            #     recording_source.name + '.depth.mp4')
            self._cv_color_cap = cv2.VideoCapture(str(recording_source))
            # self._cv_depth_cap = cv2.VideoCapture(str(depth_frame_path))

        self.color_frame_writer = self.depth_frame_writer = None
        if recording_dest is not None:
            cam_config_path = recording_dest.with_name(
                recording_dest.name + '.config.json')
            with open(cam_config_path, 'w', encoding='utf8') as cam_config_file:
                json.dump(self.active_cam_config, cam_config_file)

            # Note that using this codec requires video files to have a .mp4 extension, otherwise
            # writing frames will fail silently.
            codec = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
            color_frame_path = recording_dest.with_name(
                recording_dest.name + '.color.mp4')
            self.color_frame_writer = cv2.VideoWriter(str(color_frame_path), codec,
                                                      self.active_cam_config['frame_rate'],
                                                      self.active_cam_config['capture_res'])
            # RealSense depth frame writer commented out
            # if self._rs_pipeline is not None:
            #     depth_frame_path = recording_dest.with_name(
            #         recording_dest.name + '.depth.mp4')
            #     self.depth_frame_writer = cv2.VideoWriter(str(depth_frame_path), codec,
            #                                               self.active_cam_config['frame_rate'],
            #                                               self.active_cam_config['capture_res'])

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.recording_source is not None:
            # Read from recording source
            ret, color_image = self._cv_color_cap.read()
            if not ret:
                color_image = None

            if self._cv_depth_cap is not None:
                ret, depth_image = self._cv_depth_cap.read()
                if not ret:
                    depth_image = None
            else:
                depth_image = None

        # RealSense frame reading commented out
        # elif self._rs_pipeline is not None:
        #     frames = self._rs_pipeline.wait_for_frames()
        #
        #     if self._rs_frame_aligner is not None:
        #         frames = self._rs_frame_aligner.process(frames)
        #
        #     # depth_frame = frames.get_depth_frame()
        #     color_frame = frames.get_color_frame()
        #     if color_frame:
        #         color_image = np.asanyarray(color_frame.get_data())
        #     else:
        #         color_image = None
        #
        #     depth_frame = frames.get_depth_frame()
        #     if depth_frame:
        #         depth_image = np.asanyarray(depth_frame.get_data())
        #     else:
        #         depth_image = None

        if self.hik_frame_init is not None:
            color_image = read_hik_frame(self.hik_frame_init)
            depth_image = None

        elif self._cv_color_cap is not None:
            ret, color_image = self._cv_color_cap.read()
            if not ret:
                color_image = None

            if self._cv_depth_cap is None:
                depth_image = None
            else:
                ret, depth_image = self._cv_depth_cap.read()
                if ret:
                    B, G, R = cv2.split(depth_image)
                    depth_image = (B.astype(np.uint16) << 8) + \
                                  (G.astype(np.uint16) << 12) + R.astype(np.uint16)
                else:
                    depth_image = None


        else:
            raise RuntimeError('No image source available')

        # test = cv2.split(cv2.resize(color_image, RS_DEPTH_CAPTURE_RES))
        # depth_image = (test[0].astype(np.uint16) << 8) + test[2].astype(np.uint16)

        if self.color_frame_writer is not None and color_image is not None:
            self.color_frame_writer.write(color_image)

        if self.depth_frame_writer is not None and depth_image is not None:
            storage_format_image = cv2.merge(
                [(depth_image >> 8).astype(np.uint8), ((depth_image >> 12) % 16).astype(np.uint8),
                 (depth_image % 16).astype(np.uint8)])
            self.depth_frame_writer.write(storage_format_image)

        return color_image, depth_image

    def __del__(self):
        # RealSense pipeline cleanup commented out
        # if self._rs_pipeline is not None:
        #     self._rs_pipeline.stop()

        if self.color_frame_writer is not None:
            self.color_frame_writer.release()

        if self.depth_frame_writer is not None:
            self.depth_frame_writer.release()

        # if self.hik_frame_init is not None:
        #     hik_close(self.hik_frame_init)
