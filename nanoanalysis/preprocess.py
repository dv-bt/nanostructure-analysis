"""
This module contains functions used for image pre-processing and preliminary
operations.

Classes
-------
Baseline : the fitted surface baseline of an image

Functions
---------
trim_infoline : remove image infoline and detect pixel size
baseline_detect : detect baseline of nanostructures
baseline_import : import baseline from segmentation data
straighten_image : straighten image using given baseline as reference
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytesseract
from skimage import transform
from skimage import util
from skimage import feature
from scipy import signal
from scipy.stats import linregress


@dataclass
class Baseline:
    """
    Perform a linear regression on baseline data and store the results in a
    convenient form.

    Arguments
    ---------
    x_data : np.ndarray
        The x values to fit
    y_data : np.ndarray
        The y values to fit. It must have the same length of x_data

    Attributes
    ---------------------
    slope : float
        The baseline slope.
    intercept : float
        The baseline intercept.
    angle : float
        The baseline angle, in degrees.

    Methods
    -------
    evaluate
        Evaluate the baseline over an array representing the x-axis values.
    """
    slope: float
    intercept: float
    angle: float

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """ Calculate regression and assign field values """
        regression = linregress(x_data, y_data)
        self.slope: float = regression.slope
        self.intercept: float = regression.intercept
        self.angle = np.rad2deg(np.arctan(self.slope))

    def evaluate(self, x_array: np.ndarray) -> np.ndarray:
        """ Return the baseline y values for a given x array """
        y_array = self.slope * x_array + self.intercept
        return y_array


def trim_infoline(
    image: np.ndarray, detect_px_size: bool = True
) -> tuple[np.ndarray, float]:
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


def baseline_detect(
    image: np.ndarray, sigma: float | int = 3, num_pieces: int = 2
) -> Baseline:
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
    baseline : Baseline
        The detected baseline, a instance of the Baseline class.
    """

    edges = feature.canny(image, sigma=sigma)

    # Using piecewise gradient of edges
    edges_split = np.array_split(edges, num_pieces, axis=1)
    x_baseline = []
    y_baseline = []

    for i in range(num_pieces):
        edges_mean = signal.medfilt(
            np.mean(edges_split[i][:-1,:], axis=1), kernel_size=9
        )  # Remove last row to avoid spurious gradients
        edges_gradient = np.gradient(edges_mean)
        y_baseline.append(np.argmax(abs(edges_gradient)))
        x_baseline.append(
            image.shape[1] * (0.5 / num_pieces + i / num_pieces)
        )

    if num_pieces == 1:
        y_baseline.append(y_baseline[0])
        x_baseline.append(x_baseline[0] + 1)

    baseline = Baseline(x_baseline, y_baseline)

    return baseline


def baseline_import(
    data: pd.DataFrame, label_name: str = 'Baseline'
) -> Baseline:
    """
    Import baseline from segmentation data
    """
    data = data.loc[data.label==label_name].iloc[0]

    baseline = Baseline(data.segmentation['x'], data.segmentation['y'])

    return baseline


def straighten_image(
    image: np.ndarray, baseline: Baseline, trim_baseline: bool = True
) -> tuple[np.ndarray, float]:
    """
    Rotate image so that the input surface baseline is a horizontal line.
    The image is cropped to remove empty pixels, and its scale is preserved.

    Parameters
    ----------
    image : np.array
        Image to be rotated.
    baseline : Baseline
        The baseline around which the image has to be rotated.
    trim_baseline : bool
        Flag for cropping away everything below the detected baseline. Useful
        to simplify analysis. (default=True).


    Returns
    -------
    image_crop : np.ndarray
        Rotated and cropped image.
    baseline_val : float
        Intercept of the (horizontal) baseline on the transformed image.
    """

    angle = baseline.angle

    image_rot = transform.rotate(image, angle=angle, resize=True)
    width_crop = round(image.shape[0] * np.sin(np.deg2rad(abs(angle))) + 0.5)
    height_crop = round(image.shape[1] * np.sin(np.deg2rad(abs(angle))) + 0.5)
    image_crop = util.crop(
        image_rot,
        (
            (height_crop, height_crop), (width_crop, width_crop)
        )
    )

    baseline_val = round(
        baseline.intercept * np.cos(np.deg2rad(abs(angle))) + 0.5
    )
    if angle < 0:
        baseline_val -= height_crop

    if trim_baseline:
        image_crop = image_crop[:round(baseline_val), :]

    return image_crop, baseline_val
