"""
This module contains functions used for image pre-processing and preliminary
operations.

Functions
---------
trim_infoline : remove image infoline and detect pixel size
baseline_detect : detect baseline of nanostructures
crop_rotate : rotate image and crop unused regions
"""

import re
import numpy as np
import pytesseract
from skimage import transform
from skimage import util
from skimage import feature
from scipy import signal
from scipy.stats import linregress

def trim_infoline(image, detect_px_size=True) -> tuple[np.array, float]:
    """
    Trim an image of its infoline. Optionally, detect image pixel size by OCR,
    using Tesseract.

    Parameters
    ----------
    image : np.array
        Image to be trimmed
    detect_px_size : bool
        Flag to perform OCR of image pixel size (default=True)

    Returns
    -------
    image_trim : np.array
        Image with infoline trimmed
    px_size : float
        Image pixel size. Returns np.nan if OCR recognition is not performed
    """

    # Detect and trim infoline
    row_gradient = np.gradient(np.mean(image, axis=1))
    infoline_pos = np.argmax(abs(row_gradient))
    image_trim = image[:infoline_pos, :]

    if not detect_px_size:
        return image_trim, np.nan

    # Perform OCR recognition
    infoline = image[infoline_pos:, :]
    rescaled = util.img_as_uint(
        transform.rescale(infoline, 5, anti_aliasing=True)
    )
    ocr_text = pytesseract.image_to_string(rescaled)
    px_size = float(
        re.findall("ImagePixelSize=(.*)nm", ocr_text.replace(' ', ''))[0]
    )

    return image_trim, px_size

def baseline_detect(image, sigma=3, num_pieces=2) -> tuple[float, float]:
    """
    Detect baseline of nanostructures using maximum of piecewise gradient of
    edges.

    Parameters
    ----------
    image : np.array
        Image to be analysed.
    sigma : float
        Standard deviation of the Gaussian filter used for Canny edge filter.
        Decrease to preserve more edges. (default=3)
    num_pieces : int
        Number of pieces into which the image is divided for baseline
        detection. (default=2)

    Returns
    -------
    slope : float
        Slope of the detected baseline
    intercept : float
        Intercept of the detected baseline
    """

    edges = feature.canny(image, sigma=sigma)

    # Using piecewise gradient of edges
    edges_split = np.array_split(edges, num_pieces, axis=1)
    x_baseline = []
    y_baseline = []

    for i in range(num_pieces):
        edges_mean = signal.medfilt(
            np.mean(edges_split[i], axis=1), kernel_size=9
        )
        edges_gradient = np.gradient(edges_mean)
        y_baseline.append(np.argmax(abs(edges_gradient)))
        x_baseline.append(
            image.shape[1] * (0.5 / num_pieces + i / num_pieces)
        )

    if num_pieces == 1:
        y_baseline.append(y_baseline[0])
        x_baseline.append(x_baseline[0] + 1)

    baseline = linregress(x_baseline, y_baseline)

    return baseline.slope, baseline.intercept

def crop_rotate(
    image, angle, trim_baseline=True, baseline_val=None
) -> tuple[np.array, float]:
    """
    Rotate image, cropping it to remove empty pixels. Image scale is preserved.

    Parameters
    ----------
    image : np.array
        Image to be rotated.
    angle : float
        Rotation angle, defined as counter-clockwise from the x-axis.
    trim_baseline : bool
        Flag for cropping away everything below the detected baseline. Useful
        to simplify analysis. (default=True).
    baseline_val = None or float
        Position of the horizontal final baseline.


    Returns
    -------
    image_crop : np.array
        Rotated and cropped image.
    baseline_val : float
        Intercept of the (horizontal) baseline.
    """

    image_rot = transform.rotate(image, angle=angle, resize=True)
    width_crop = round(image.shape[0] * np.sin(np.deg2rad(abs(angle))) + 0.5)
    height_crop = round(image.shape[1] * np.sin(np.deg2rad(abs(angle))) + 0.5)
    image_crop = util.crop(
        image_rot,
        (
            (height_crop, height_crop), (width_crop, width_crop)
        )
    )

    if trim_baseline:
        if not baseline_val:
            _, baseline_val = baseline_detect(image_crop, num_pieces=1)
        image_crop = image_crop[:round(baseline_val), :]

    return image_crop, baseline_val
