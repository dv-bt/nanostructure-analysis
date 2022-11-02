# nanoanalysis

A Python package to analyse instances of rod-like nanostructures.

## Structure

```
nanoanalysis
├── __init__.py
├── README.md
├── preprocess.py
└── segmentation.py
```

The `preprocess` module contains functions used for image preprocessing. The `segmentation` module contains functions for importing and analysing segmentation masks obtained via third-party software.

## Usage

This package provides a series of functions to analyse instances of rod-like nanostructures from experimental Scanning Electron Microscopy side-view images of surfaces. The package is broken down in the `preprocess` and `segmentation` modules.

### preprocess

This module deals with common preprocessing tasks, such as image rotation and surface baseline detection.

- `trim_infoline`: SEM pictures often have an infoline at the bottom of the image, containing acquisition metadata. This function performs a naive detection by finding the absolute maximum (or minimum) of the image gradient, averaged by row, and returns the image without the infoline. Optionally, the image pixel size in nm can be extracted by OCR.

- `Baseline`: a representation of the surface baseline, calculated from a linear regression of supplied data points.
  
- `baseline_detect`: the surface baseline is detected by finding the maximum edge gradient over a set of `num_pieces` vertical partitions of the image, and stored as a `Baseline` instance.

- `baseline_import`: import an externally-determined baseline from segmentation data and store it as a `Baseline` instance. The baseline data must be present in the database imported with `segmentation.import_segmentation`.

- `straighten_image`: straighten an image using its surface baseline as a reference; in the transformed image, the baseline appears as a horizontal line, simplifying the subsequent analysis. This function crops away the unused black-pixel regions resulting from image rotation and preserves image scale. By default, the unused image region below the baseline is also cropped.

### segmentation

This module deals with the import and analysis of instance segmentation masks.

- `import_segmentation`: import segmentation masks. At the moment, supported database formats are CVAT for images 1.1 (default) and COCO 1.0.

- `rod_analysis`: analyse an instance of a rod-like nanostructure, taking as additional inputs the original image pixel size in nm and its surface baseline. The rod skeleton is detected using the Lee method and analysed with [skan](https://github.com/jni/skan). The rod diameter is estimated for each point of the skeleton from the distance transform, and its volume is approximated as the sum of a series truncated cones.

## Requirements

This package requires an installation of Python 3.7 or newer.

Additionally, the following packages are required:
- Numpy 1.23.3
- Pandas 1.5
- Scikit-image 0.19.3
- Scikit-learn 1.1.2
- Scipy 1.9.1
- Skan 0.10