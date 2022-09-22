"""
This module contains functions used for image pre-processing.

Functions
---------
trim_infoline : remove image infoline and detect pixel size
"""

import re
import numpy as np
import pytesseract
from skimage import transform
from skimage import util

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
