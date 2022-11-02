"""
This script analyses the shape of nanorods from SEM side-view images.

Arguments
---------
-p --plot_images
    Plot reconstructed rods on original image
-n --new_files
    Only analyse not previously analyed images
-v --verbose
    Enable verbose output
"""

from os.path import exists
from glob import glob
from pathlib import Path
import argparse

import pandas as pd
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nanoanalysis import preprocess
from nanoanalysis import segmentation


# Build argumet parser
ap = argparse.ArgumentParser()
ap.add_argument(
    '-p', '--plot_images',
    required=False, action='store_true',
    help="Plot reconstructed rods for visual validation"
)
ap.add_argument(
    '-n', '--new_files',
    required=False, action='store_true',
    help="Only analyse not previously analysed images"
)
ap.add_argument(
    '-v', '--verbose',
    required=False, action='store_true',
    help="Enable verbose output"
)
args = vars(ap.parse_args())

# Script variables
IMAGE_FOLDER = "Data/Images/"
SEGMENTATION_FOLDER = "Data/Segmentation-masks/"
RESULTS_FOLDER =  "Data/Results/"

Path(RESULTS_FOLDER).mkdir(exist_ok=True)

data = segmentation.import_segmentation(SEGMENTATION_FOLDER)

for image_name in tqdm(data.file_name.unique()):

    if args['verbose']:
        print('Current image:', image_name)

    results_path = RESULTS_FOLDER + image_name.replace('tif', 'csv')
    if args['new_files'] and exists(results_path):
        continue

    data_image = data.query("file_name==@image_name")

    rod_list = []

    try:
        image_path = glob(IMAGE_FOLDER + "/**/" + image_name)[0]

        # Load image
        image_read = io.imread(image_path)

        # Remove infoline from image.
        # This passage can be skipped if the image does not contain any
        # infoline. Pixel size should then be specified manually or inferred
        # from formatted file name.
        image, px_size = preprocess.trim_infoline(image_read)

        # Detect surface and correct for image tilt. Skip if baseline info is
        # present in segmentation data and use that instead.
        baseline = (
            preprocess.baseline_detect(image, num_pieces=2)
            if not data_image.label.isin(['Baseline']).any()
            else preprocess.baseline_import(data_image)
        )

        image_rot, _ = preprocess.straighten_image(
            image, baseline, trim_baseline=True
        )

        if args['plot_images']:
            fig, ax = plt.subplots()
            ax.imshow(image_rot, cmap='gray')
            ax.set_title(image_name)

        for i in data_image.query("label!='Baseline'").index:
            rods = (
                segmentation.rod_analysis(
                    data_image.loc[i], px_size, baseline
                )
                .assign(**{'file_name': image_name})
            )
            rod_list.append(rods)

            if args['plot_images']:
                for _, rod in rods.groupby('instance_id'):
                    line = ax.plot(rod.x, rod.y)
                    for i in range(0, len(rod), 10):
                        ax.add_patch(mpatches.Circle(
                            (rod.iloc[i].x, rod.iloc[i].y),
                            rod.iloc[i].dist_transform,
                            color=line[0].get_color(), alpha=0.3
                        ))

        rods = pd.concat(rod_list)
        rods = rods.drop(['dist_transform'], axis=1)
        rods.to_csv(
            "Data/Results/" + image_name.replace('tif', 'csv'), index=False
        )

        if args['plot_images']:
            plt.show()
    except ValueError:
        print('ERROR:', image_name, 'not analysed.')
