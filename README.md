# nanostructure-analysis
A Python library for the analysis of rod-like nanostructures from SEM images.

## Usage

The [nanostructure-analysis.py](nanostructure_analysis.py) script provides an example of a full analysis of experimental Scanning Electron Microscopy (SEM) images to extract nanostructure data using the code provided in the `nanoanalysis` package. Further information on the latter can be found in the [package readme](nanoanalysis/README.md).

The script can be readily adapted to analyse any appropriate dataset by supplying the paths to the folders storing images, segmentation masks, and results, which are controlled, respectively, by the `IMAGE_FOLDER`, `SEGMENTATION_FOLDER`, and `RESULTS_FOLDER` variables.

Instance segmentation data should be obtained via external software, and supplied in one of the supported formats. Currently, the CVAT for images 1.1 and COCO 1.0 formats are supported.

## Requirements

The code in the repository requires a Python 3.7 or newer installation, which must include Numpy, Pandas, Scipy, Scikit-Image, and Scikit-Learn. Skeleton analysis relies on [skan](https://github.com/jni/skan). Optional plotting functions require matplotlib, and optional metadata extraction by OCR relies on PyTesseract.

A complete list of the packages in the virtual environment used for the development of this code is included in [environment.yml](environment.yml).