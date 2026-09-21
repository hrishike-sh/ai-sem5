"""Conservative, deterministic IVP recommendations. No learned models."""
import cv2
import numpy as np

from .pipeline import Settings, quality


def choose_settings(rgb):
    small = cv2.resize(rgb, (min(640, rgb.shape[1]), max(2, round(rgb.shape[0] * min(640, rgb.shape[1]) / rgb.shape[1]))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    measured = quality(small)
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    noise = float(np.median(np.abs(residual)))
    gamma = 1.5 if measured['brightness'] < 65 else .85 if measured['brightness'] > 205 else 1.
    # CLAHE can undo bright-input gamma reduction and clip low-contrast detail,
    # especially when small tiles contain only a narrow intensity range.
    clahe = gamma >= 1 and measured['contrast'] < 40 and measured['highlight_fraction'] < .15
    # Never invent detail with automatic sharpening or shift unknown scene colours.
    denoise = 'Median' if noise >= 5 else 'Gaussian'
    settings = Settings(max_width=1280, gamma=gamma, clahe=clahe, clip_limit=1.5,
                        denoise=denoise, kernel=3, canny_low=45, canny_high=130)
    reasons = [f"Analysis capped at 1280 px wide; aspect ratio and source-coordinate mapping are retained.",
               f"{denoise} 3×3 smoothing: {'median residual suggests substantial fine variation' if noise >= 5 else 'mild smoothing before feature extraction'}.",
               f"Gamma {gamma:g}: {'dark input' if gamma > 1 else 'bright input' if gamma < 1 else 'brightness is within the conservative range'}.",
               f"Colour CLAHE {'enabled for low contrast' if clahe else 'left off for bright input to avoid undoing gamma reduction' if gamma < 1 else 'left off'}.",
               "The separate grayscale analysis branch always demonstrates CLAHE before extracting edges and binary masks.",
               "No automatic sharpening, saturation shift, or white-balance correction."]
    return settings, {'method': 'deterministic_ivp_v2', 'measurements': measured,
                      'median_residual': noise, 'reasons': reasons,
                      'limitation': 'Heuristics are a starting point, not an optimum or an accuracy guarantee.'}
