"""
Blue LED detector for opponent robot targeting.
Draws a circle on the full camera feed around the brightest blue light.

Based on scene analysis:
  - Target: bright emissive blue LEDs on robot undercarriage
  - Competing sources: fluorescent ceiling lights, exit sign, glass reflections
  - Strategy: tight HSV hue for blue + high saturation to reject white/gray lights
               + minimum brightness to reject faint ambient blue tints
               + env suppression ONLY if image_env.png is a no-lights background photo

Controls:
  q         — quit
  d         — toggle live HSV diagnostics in terminal
  +/-       — raise/lower VAL_MIN brightness threshold
  s/S       — lower/raise SAT_MIN saturation threshold
  e         — toggle env suppression on/off (to compare)
"""

import os, sys, time, ctypes
import cv2
import numpy as np
from pathlib import Path

# ── SDK paths ──────────────────────────────────────────────────────────────────
WIN_DLL_DIRS = [
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
]
if sys.platform == "win32":
    dll_dir = next((d for d in WIN_DLL_DIRS
                    if os.path.exists(os.path.join(d, "MvCameraControl.dll"))), None)
    if dll_dir is None:
        raise FileNotFoundError("MvCameraControl.dll not found.")
    os.add_dll_directory(dll_dir)
    os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

from MVS.MvCameraControl_class import *

# ── Filter parameters — tuned for bright emissive blue LEDs ───────────────────
# Hue 100–130 = pure blue in OpenCV HSV (0–180 scale).
# The robot LEDs in the image are saturated royal/cobalt blue → fits here.
# The exit sign is greenish (lower hue ~80-90) → rejected by HUE_MIN=100.
# Ceiling lights are white (low saturation) → rejected by SAT_MIN.
HUE_MIN       = 100
HUE_MAX       = 130

# High saturation rejects white ceiling lights and gray/glass reflections.
# The emissive blue LEDs on the robot are deeply saturated → stay well above 120.
SAT_MIN       = 120

# Brightness floor. Start at 100 — LEDs are bright so they'll be well above this.
# Raise with '+' to filter false positives, lower with '-' if LEDs aren't detected.
VAL_MIN       = 100

# Minimum blob area in pixels — rejects single-pixel noise
MIN_BLOB_AREA = 15

# Env suppression: pixel must be ENV_V_DELTA brighter in V than the env image.
# NOTE: only works correctly if image_env.png was captured WITHOUT the robot LEDs on.
# If your env image has the LEDs on, press 'e' to disable this at runtime.
ENV_V_DELTA   = 15
USE_ENV       = True    # toggled at runtime with 'e' key

# ── Camera settings ────────────────────────────────────────────────────────────
# Fixed exposure is critical — auto-exposure adapts to dark arena and dims LEDs
# unpredictably, making thresholds unreliable.
# Tune: if image is too dark → increase EXPOSURE_US or GAIN_DB
#        if LEDs blow out to white → decrease EXPOSURE_US
EXPOSURE_US   = 4000.0   # microseconds
GAIN_DB       = 8.0      # dB
FPS           = 120.0

IMAGE_ENV_NAME = "image_env.png"


def cam_config(cam, stDevInfo):
    cam.MV_CC_CreateHandle(stDevInfo)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)

    # Disable auto exposure — the single most important fix for consistent detection
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    cam.MV_CC_SetFloatValue("ExposureTime", EXPOSURE_US)

    cam.MV_CC_SetEnumValue("GainAuto", 0)
    cam.MV_CC_SetFloatValue("Gain", GAIN_DB)

    # White balance once at startup, not continuous
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 1)

    cam.MV_CC_SetEnumValue("TriggerMode", 0)
    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
    cam.MV_CC_SetFloatValue("AcquisitionFrameRate", FPS)

    cam.MV_CC_StartGrabbing()


def load_env(script_dir: Path):
    path = script_dir / IMAGE_ENV_NAME
    if not path.exists():
        print(f"[WARN] {IMAGE_ENV_NAME} not found at {path} — env suppression disabled.")
        return None
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Could not read {path} — env suppression disabled.")
        return None
    print(f"[INFO] Loaded env image: {path}  shape={img.shape}")
    return img


def build_blue_mask(img_bgr, img_env, val_min, sat_min, use_env):
    """
    Returns binary mask of blue LED pixels and the HSV image.

    Rejection logic:
      - White/gray (ceiling fluorescents)  → low saturation → killed by SAT_MIN
      - Exit sign (greenish-blue)          → hue ~85-95    → killed by HUE_MIN=100
      - Glass/floor reflections (faint)    → low V         → killed by VAL_MIN
      - Env suppression (if enabled)       → not brighter than background by ENV_V_DELTA
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([HUE_MIN, sat_min, val_min])
    upper = np.array([HUE_MAX, 255,     255])
    mask  = cv2.inRange(hsv, lower, upper)

    # Env suppression: only keep pixels brighter than the background env image
    if use_env and img_env is not None:
        env_resized = img_env
        if img_env.shape[:2] != img_bgr.shape[:2]:
            env_resized = cv2.resize(img_env, (img_bgr.shape[1], img_bgr.shape[0]),
                                     interpolation=cv2.INTER_AREA)
        v_cur = hsv[:, :, 2].astype(np.int16)
        v_env = cv2.cvtColor(env_resized, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.int16)
        brighter = ((v_cur - v_env) > ENV_V_DELTA).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, brighter)

    # Morphology: open removes speckle noise, close fills holes inside blobs
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    return mask, hsv


def detect_blobs(mask, hsv):
    """
    Find contours → blobs, sorted brightest first.
    Returns list of (cx, cy, radius, mean_V, area).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx     = int(M["m10"] / M["m00"])
        cy     = int(M["m01"] / M["m00"])
        radius = max(12, int(np.sqrt(area / np.pi)) + 8)

        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        mean_v = cv2.mean(hsv[:, :, 2], mask=blob_mask)[0]

        blobs.append((cx, cy, radius, mean_v, area))

    blobs.sort(key=lambda b: b[3], reverse=True)
    return blobs


def draw_results(img_bgr, blobs, fps, val_min, sat_min, use_env):
    """Full camera feed with detection circles overlaid."""
    out = img_bgr.copy()

    for i, (cx, cy, radius, mean_v, area) in enumerate(blobs):
        if i == 0:
            # Primary target — bright green double circle
            cv2.circle(out, (cx, cy), radius,     (0, 255, 0), 3)
            cv2.circle(out, (cx, cy), radius + 5, (0, 255, 0), 1)
            cv2.circle(out, (cx, cy), 4,           (255, 255, 255), -1)
            cv2.putText(out, f"TARGET  V={mean_v:.0f}",
                        (cx + radius + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            # Secondary blobs — thin cyan circle
            cv2.circle(out, (cx, cy), radius, (0, 200, 200), 1)
            cv2.putText(out, f"#{i+1}",
                        (cx + radius + 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

    env_txt = "ENV:ON" if use_env else "ENV:OFF"
    hud = f"FPS:{fps:.1f}  VAL:{val_min}  SAT:{sat_min}  {env_txt}  Blobs:{len(blobs)}"
    cv2.putText(out, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(out, "+/-:brightness  s/S:saturation  e:env  d:diag  q:quit",
                (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return out


def print_diagnostics(img_bgr, blobs, val_min, sat_min):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    print(f"  VAL_MIN={val_min}  SAT_MIN={sat_min}")
    print(f"  Frame V — min:{v_ch.min():3d}  max:{v_ch.max():3d}  mean:{v_ch.mean():.1f}")
    print(f"  Frame S — min:{s_ch.min():3d}  max:{s_ch.max():3d}")
    if blobs:
        cx, cy, r, mv, area = blobs[0]
        ph = int(hsv[cy, cx, 0])
        ps = int(hsv[cy, cx, 1])
        pv = int(hsv[cy, cx, 2])
        print(f"  Best blob center ({cx},{cy}) — H:{ph}  S:{ps}  V:{pv}  "
              f"mean_V:{mv:.1f}  area:{area:.0f}px")
    else:
        print("  No blobs found — press '-' to lower VAL_MIN or 's' to lower SAT_MIN")
    print()


def main():
    val_min = VAL_MIN
    sat_min = SAT_MIN
    use_env = USE_ENV

    script_dir = Path(__file__).resolve().parent
    img_env    = load_env(script_dir)
    if img_env is None:
        use_env = False

    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        raise RuntimeError("No HIK camera found.")

    stDevInfo   = ctypes.cast(deviceList.pDeviceInfo[0],
                               ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    cam = MvCamera()
    cam_config(cam, stDevInfo)

    payload = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", payload)
    payload_size = int(payload.nCurValue)
    data_buf     = (ctypes.c_ubyte * payload_size)()

    diag_mode = False
    last_diag = 0.0
    t_prev    = time.perf_counter()

    print("Ready. Controls: q=quit  d=diagnostics  +/-=VAL_MIN  s/S=SAT_MIN  e=env toggle")
    print(f"Starting: VAL_MIN={val_min}  SAT_MIN={sat_min}  "
          f"EXPOSURE={EXPOSURE_US}us  GAIN={GAIN_DB}dB  ENV={'ON' if use_env else 'OFF'}")

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, stFrameInfo, 1000)
        if ret != 0:
            continue

        w = stFrameInfo.nWidth
        h = stFrameInfo.nHeight
        img_rgb = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        mask, hsv = build_blue_mask(img_bgr, img_env, val_min, sat_min, use_env)
        blobs     = detect_blobs(mask, hsv)

        t_now  = time.perf_counter()
        fps    = 1.0 / (t_now - t_prev) if t_now > t_prev else 0.0
        t_prev = t_now

        out = draw_results(img_bgr, blobs, fps, val_min, sat_min, use_env)
        cv2.imshow("Blue LED Detector", out)

        if diag_mode and (t_now - last_diag) >= 0.5:
            print_diagnostics(img_bgr, blobs, val_min, sat_min)
            last_diag = t_now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            diag_mode = not diag_mode
            print(f"Diagnostics {'ON' if diag_mode else 'OFF'}")
        elif key in (ord("+"), ord("=")):
            val_min = min(255, val_min + 5)
            print(f"VAL_MIN → {val_min}")
        elif key == ord("-"):
            val_min = max(0, val_min - 5)
            print(f"VAL_MIN → {val_min}")
        elif key == ord("S"):
            sat_min = min(255, sat_min + 5)
            print(f"SAT_MIN → {sat_min}")
        elif key == ord("s"):
            sat_min = max(0, sat_min - 5)
            print(f"SAT_MIN → {sat_min}")
        elif key == ord("e"):
            if img_env is None:
                print("Cannot enable env suppression — image_env.png failed to load.")
            else:
                use_env = not use_env
                print(f"Env suppression {'ON' if use_env else 'OFF'}")

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()