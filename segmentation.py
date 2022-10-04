"""
This module contains functions used to import segmentation masks and process
object instances.

Functions
---------
- import_segmentation : import segmentation masks
"""

import json
import numpy as np
import pandas as pd

def import_segmentation(file_path) -> pd.DataFrame:
    """
    Import segmentation mask stored in a JSON file follwing the the COCO 1.0
    format.

    Parameters
    ----------
    file_path : str
        Path to the JSON database

    Returns
    -------
    data : pd.DataFrame
        Pandas dataframe with the segmentation masks
    """

    def _reshape_coordinates(coords) -> dict:
        """ Reshape coordinates from flat list in COCO JSON """

        coords_array = np.array(coords).reshape(int(len(coords) / 2), 2)
        coords_dict = {
            'x': coords_array[:, 0],
            'y': coords_array[:, 1]
        }

        return coords_dict

    with open(file_path, 'r') as file:
        data_json = json.load(file)

    images = pd.json_normalize(data_json['images'])
    images.rename(columns={'id': 'image_id'}, inplace=True)
    images.drop(
        ['license', 'flickr_url', 'coco_url', 'date_captured'],
        axis=1, inplace=True
    )

    annotations = pd.json_normalize(data_json['annotations'])
    annotations['segmentation'] = (
        annotations
        .segmentation
        .apply(lambda x: _reshape_coordinates(x[0]))
    )
    annotations.drop(
        ['iscrowd', 'area', 'attributes.occluded', 'bbox'],
        axis=1, inplace=True
    )

    data = pd.merge(images, annotations)

    return data
