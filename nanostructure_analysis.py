"""
This script analyses the shape of nanorods from SEM side-view
images in the /Images directory.
"""

import pandas as pd
from skimage import io
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import preprocess
import segmentation
import utility


data = segmentation.import_segmentation('Images/Training-masks')

rod_list = []

for image_name in data.file_name.unique():

    # Load image
    image_read = io.imread("Images/" + image_name)

    # Remove infoline from image
    # Note: this only works for images similar to those analysed here!
    image, px_size = preprocess.trim_infoline(image_read)
    baseline_angle, _ = preprocess.baseline_detect(image, num_pieces=2)
    image_rot, baseline_val = preprocess.crop_rotate(
        image, baseline_angle, trim_baseline=True
    )

    for i in data.query("file_name==@image_name").index:
        rod_list.append(
            segmentation.rod_analysis(
                data.loc[i], px_size, baseline_angle, baseline_val
            )
        )

    # fig, ax = plt.subplots()
    # ax.imshow(image_rot, cmap='gray')
    # for _, rod in rods.groupby('instance_id'):
    #     line = ax.plot(rod.x, rod.y)
    #     for i in range(0, len(rod), 10):
    #         ax.add_patch(mpatches.Circle(
    #             (rod.iloc[i].x, rod.iloc[i].y), rod.iloc[i].dist_transform,
    #             color=line[0].get_color(), alpha=0.3
    #         ))
    # plt.show()

rods = pd.concat(rod_list)
