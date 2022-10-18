"""
This script analyses the shape of nanorods from SEM side-view images
"""

from glob import glob
import pandas as pd
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import preprocess
import segmentation


data = segmentation.import_segmentation('Data/Segmentation-masks')
VALIDATE = False

rod_list = []

for image_name in tqdm(data.file_name.unique()):

    image_path = glob("Data/Images/**/" + image_name)[0]

    # Load image
    image_read = io.imread(image_path)

    # Remove infoline from image
    # Note: this only works for images similar to those analysed here!
    image, px_size = preprocess.trim_infoline(image_read)
    baseline_angle, _ = preprocess.baseline_detect(image, num_pieces=2)
    image_rot, baseline_val = preprocess.crop_rotate(
        image, baseline_angle, trim_baseline=True
    )

    if VALIDATE:
        fig, ax = plt.subplots()
        ax.imshow(image_rot, cmap='gray')

    for i in data.query("file_name==@image_name").index:
        rods = segmentation.rod_analysis(
            data.loc[i], px_size, baseline_angle, baseline_val
        ).assign(**{'file_name': image_name})
        rod_list.append(rods)

        if VALIDATE:
            for _, rod in rods.groupby('instance_id'):
                line = ax.plot(rod.x, rod.y)
                for i in range(0, len(rod), 10):
                    ax.add_patch(mpatches.Circle(
                        (rod.iloc[i].x, rod.iloc[i].y),
                        rod.iloc[i].dist_transform,
                        color=line[0].get_color(), alpha=0.3
                    ))

    if VALIDATE:
        plt.show()

rods = pd.concat(rod_list)
rods = rods.drop(['dist_transform'], axis=1)
rods.to_csv("Data/rod-database.csv", index=False)
