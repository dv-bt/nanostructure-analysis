"""
This module contains functions used to import segmentation masks and process
object instances.

Functions
---------
- import_segmentation : import segmentation masks.
- rod_analysis : analyse rod height, diameter, and volume.
"""

import os
import glob
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy import ndimage
from skimage import draw
from skimage import morphology
from skimage.util import img_as_bool
import skan

from . import preprocess


def import_segmentation(
    folder_path: str, database_type: str = 'CVAT'
) -> pd.DataFrame:
    """
    Import segmentation mask stored XML or JSON databases.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the files.
    database_type : str
        Format of segmentation mask database. Currently supported values:
        - 'COCO': COCO 1.0 JSON
        - 'CVAT': CVAT for images 1.1
        (default='CVAT')

    Returns
    -------
    data : pd.DataFrame
        Pandas dataframe with the segmentation masks.
    """

    # Check function arguments
    if database_type not in {'COCO', 'CVAT'}:
        raise ValueError(
            'Value of the database_type argument is not valid.'
            'Allowed arguments: "CVAT", "COCO"'
        )

    data_list = []

    if not folder_path.endswith(os.sep):
        folder_path += os.sep

    search_path = folder_path + '*.' + (
        'json' if database_type=='COCO' else 'xml'
    )

    for file_path in glob.glob(search_path):
        data_list.append(
            __read_coco(file_path) if database_type=='COCO'
            else __read_cvat(file_path)
        )

    data = pd.concat(data_list, ignore_index=True)

    data['segmentation'] = (
        data
        .segmentation
        .apply(__reshape_coordinates)
    )

    data['instance_id'] = 1
    data['instance_id'] = data.groupby(
        ["file_name", "label"]
    ).instance_id.cumsum()
    data = data.astype({'width': int, 'height': int})

    return data


def __reshape_coordinates(coords: list) -> dict:
    """ Reshape coordinates from flat list """

    coords_array = np.array(coords).reshape(int(len(coords) / 2), 2)
    coords_dict = {
        'x': coords_array[:, 0],
        'y': coords_array[:, 1]
    }

    return coords_dict


def __read_coco(file_path: str) -> pd.DataFrame:
    """ Read COCO 1.0 JSON annotation file """
    with open(file_path, 'r', encoding='utf8') as file:
        data_json = json.load(file)

    images = pd.json_normalize(data_json['images'])
    images = images.rename(columns={'id': 'image_id'})
    images = images.drop(
        ['license', 'flickr_url', 'coco_url', 'date_captured'],
        axis=1
    )

    categories = pd.json_normalize(data_json['categories'])
    categories = categories.drop('supercategory', axis=1)
    categories = categories.rename(
        columns={'id': 'category_id', 'name': 'label'}
    )

    annotations = pd.json_normalize(data_json['annotations'])
    annotations = annotations.drop(
        ['iscrowd', 'area', 'attributes.occluded', 'bbox', 'id'],
        axis=1
    )
    annotations = (
        pd.merge(images, pd.merge(categories, annotations))
        .drop(['image_id', 'category_id'], axis=1)
    )

    # Format segmentation vector
    annotations['segmentation'] = (
        annotations
        .segmentation
        .apply(lambda x: x[0])
    )

    return annotations


def __read_cvat(file_path: str) -> pd.DataFrame:
    """ Read CVAT for images 1.1 XML annotation file """

    annotation_list = []

    tree = ET.parse(file_path)
    root = tree.getroot()
    for image in root.findall('image'):
        for annotation in image:
            annotation_list.append(
                pd.DataFrame(annotation.attrib, index=[0])
                .assign(**image.attrib)
            )

    annotations = (
        pd.concat(annotation_list, ignore_index=True)
        .drop(['occluded', 'source', 'z_order', 'id'], axis=1)
        .rename(columns={'points': 'segmentation', 'name': 'file_name'})
    )

    # Format segmentation vector
    annotations['segmentation'] = (
        annotations
        .segmentation
        .apply(lambda x: np.fromstring(x.replace(';', ','), sep=','))
    )

    return annotations


def __rod_height(
    data: pd.DataFrame, drawing: np.ndarray, px_size: float
) -> pd.DataFrame:
    """
    Calculate the height of rods at each point along a given path, returning
    the input dataframe with the added data.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing rod skeleton and values of distance
        transform.
        NOTE: data should contain information about a single rod.
    drawing : np.ndarray
        Binary image of the rod, obtained from segmentation mask.
    px_size : float
        The size in nm of each image pixel.

    Return
    ------
    data_out : pd.DataFrame
        The input dataframe with the added height column
    """

    data_out = data.copy()

    data_out = __rod_bottom(data_out, drawing)
    data_out = __rod_tip(data_out, drawing)

    # Calculate Euclidean distance between consecutive points
    data_calc = data_out.copy()
    data_calc['dist_x'] = data_calc.x.diff().fillna(0)
    data_calc['dist_y'] = data_calc.y.diff().fillna(0)
    data_calc['dist'] = np.sqrt(data_calc.dist_x ** 2 + data_calc.dist_y ** 2)

    # Convert to height
    data_calc['dist_cumul'] = data_calc['dist'].cumsum()

    data_out['height'] = (
        data_calc['dist_cumul'].max() - data_calc['dist_cumul']
    ) * px_size

    return data_out


def __rod_bottom(
    data: pd.DataFrame, drawing: np.ndarray,
    fit_fraction: float = 0.05, sigma: float | int = 2
) -> pd.DataFrame:
    """
    Extrapole rod bottom to baseline and remove spurious "roots"
    """

    data_out = data.copy()
    data_calc = data.copy()

    # Handle presence of spurious "roots", by removing monotonically
    # decreasing regions at the rod base.
    data_calc['dgm'] = pd.Series(
        np.gradient(
            data_calc.dist_transform.rolling(5).mean()
        )
    ).rolling(5).mean()
    data_bottom = data_calc.tail(round(len(data_calc) * fit_fraction))

    if data_bottom.dgm.mean() <= (
        data_calc.dgm.mean() - sigma * data_calc.dgm.std()
    ):
        data_calc['dgm_sign'] = np.sign(data_calc.dgm)
        data_calc['dgm_diff'] = data_calc.dgm_sign.diff()
        ix_root = data_calc.loc[data_calc.dgm_diff==-1].index.max()

        # Remove root and redefine frames
        data_out = data_out.iloc[:ix_root].copy()
        data_bottom = data_out.tail(round(len(data_out) * fit_fraction))

    # Extrapolate rod bottom to baseline
    regression_b = linregress(data_bottom.y, data_bottom.x)
    data_out.loc[data_out.index.max() + 1, ['y', 'x']] = [
        drawing.shape[0] - 1,
        (drawing.shape[0] - 1) * regression_b.slope + regression_b.intercept
    ]

    return data_out


def __rod_tip(
    data: pd.DataFrame, drawing: np.ndarray, fit_fraction: float = 0.08
) -> pd.DataFrame:
    """
    Extrapolate tip to contour.
    """

    data_tip = data.head(round(len(data) * fit_fraction)).copy()

    regression_t = linregress(data_tip.y, data_tip.x)
    tip_array = np.zeros((int(data_tip.y.min()), 2))
    tip_array[:, 1] = np.arange(data_tip.y.min())
    tip_array[:, 0] = np.around(
        regression_t.intercept + tip_array[:, 1] * regression_t.slope
    )

    for i, row in enumerate(reversed(tip_array)):
        if drawing[int(row[1]), int(row[0])] == 0:
            ix_tip = len(tip_array) - i
            break
    data_out = pd.concat(
        [
            pd.DataFrame(
                {'x': tip_array[ix_tip][0], 'y': tip_array[ix_tip][1]},
                index=[0]
            ),
            data
        ], ignore_index=True
    )

    return data_out


def __rod_volume(data: pd.DataFrame) -> np.ndarray:
    """ Calculate rod volume by approximation as series of truncated cones """

    data_calc = data.copy()

    # Fill missing values with nearest neighbour
    data_calc['radius'] = (
        data_calc.diameter.fillna(method='bfill').fillna(method='ffill') / 2
    )
    data_calc['radius_shift'] = data_calc.radius.shift(1)
    data_calc['height_diff'] = data_calc.height.diff().abs()
    data_calc['volume'] = (
        np.pi / 3 * data_calc.height_diff * (
            data_calc.radius ** 2 +
            data_calc.radius * data_calc.radius_shift +
            data_calc.radius_shift ** 2
        )
    )

    return data_calc.volume.cumsum().to_numpy()[::-1]


def rod_analysis(
    data: pd.DataFrame, px_size: float,
    baseline_angle: float, baseline_intercept: float
) -> pd.DataFrame:
    """
    Extract rod skeleton and diameter for each point along it.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe with rod data as formatted by import_segmentation.
        NOTE: data should contain information about a single rod.
    px_size : float
        Size of each pixel in nm.
    baseline_angle : float
        Angle of the surface baseline, in degrees.
    baseline_intercept : float
        Intercept of the surface baseline.

    Returns
    -------
    results : pd.DataFrame
        Pandas dataframe with the results of the analysis.
    """

    drawing = np.zeros((data['height'], data['width']))
    polygon = draw.polygon(
        data['segmentation']['y'],
        data['segmentation']['x']
    )
    drawing[polygon] = 1
    drawing, _ = preprocess.crop_rotate(
        drawing, angle=baseline_angle, baseline_intercept=baseline_intercept
    )

    distance = ndimage.distance_transform_edt(drawing)
    skeleton = img_as_bool(morphology.skeletonize(drawing, method='lee'))
    skeleton_skan = skan.Skeleton(distance * skeleton)

    # Find main branch and merge when necessary
    branches = (
        skan.summarize(
            skeleton_skan, find_main_branch=True
        ).query("main")
    )

    path_list = []
    for path in branches.index:
        path_df = pd.DataFrame(
            skeleton_skan.path_coordinates(path), columns=['y', 'x']
        )
        path_df[['px_index', 'dist_transform']] = (
            pd.DataFrame(skeleton_skan.path_with_data(path)).T
        )
        path_list.append(path_df)

    results = pd.concat(path_list).drop_duplicates('px_index')
    results = __rod_height(results, drawing, px_size)
    results['instance_id'] = data['instance_id']
    results['diameter'] = results.dist_transform * 2 * px_size
    results['volume'] = __rod_volume(results)

    results = results.drop(['px_index'], axis=1)

    return results
