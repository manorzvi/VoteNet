import matplotlib.colors as co
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List, Optional, Tuple, Union


class PointcloudPlacement(object):

    BBOX_PLANES_LAYOUT = (
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        ((0, 1, 4, 5), (2, 3, 6, 7)),
        ((0, 2, 4, 6), (1, 3, 5, 7)),
    )

    @staticmethod
    def compute_plane_params(points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the plane's coefficients for the given points.
        The plane's eq. is: a x + b y + c z + d = 0 {dot(normal, point) + d = 0}.

        :param coords: an array of shape (m, 3) where m >= 3 (only first 3 points are used).
        :return: a tuple of the plane's (normal, bias).
        """
        v1 = points[2] - points[0]
        v2 = points[1] - points[0]
        normal = np.cross(v1, v2)
        d = -1.0 * np.dot(normal, points[0])
        return normal, d

    @staticmethod
    def above_or_below_plane(pointcloud: np.ndarray, points) -> np.ndarray:
        """
        Compute whether each point in the pointcloud is above/below a given plane.

        :param pointcloud: pointcloud of shape (num_points, 3).
        :param points: points on the plane (array of shape (m, 3), where m > 3).
        :return: array of size (num_points,) with values in {-1, 1},
                 indicating above/below plane for each point in poincloud.
        """
        normal, bias = PointcloudPlacement.compute_plane_params(points=points)
        above_or_below = np.sign(pointcloud @ normal + bias)
        return above_or_below

    @staticmethod
    def intersect_bbox(pointcloud: np.ndarray, bbox: np.ndarray, tolerance: Optional[float] = 0.1) -> bool:
        """
        Checks if a given pointcloud intersects a bounding box defines by corner points.

        :param pointcloud: pointcloud of shape (num_points, 3).
        :param bbox: corners of a bounding box, as an array of shape (8, 3)
        :param tolerance: tolerance of intersection.
        :return: bool indicating if the pointcloud intersects the bounding box.
        """

        # Assume all points in bbox
        num_points = pointcloud.shape[0]
        points_inside_cube = np.ones(num_points)

        # Iterate over 3 possible planes' pairs layouts
        for layout in PointcloudPlacement.BBOX_PLANES_LAYOUT:
            indicator_x1 = PointcloudPlacement.above_or_below_plane(pointcloud=pointcloud, points=bbox[layout[0], :])
            indicator_x2 = PointcloudPlacement.above_or_below_plane(pointcloud=pointcloud, points=bbox[layout[1], :])
            points_between_planes = (indicator_x1 * indicator_x2 <= 0).astype(np.int32)
            points_inside_cube *= points_between_planes

        num_points_in_cude = np.sum(points_inside_cube)
        return num_points_in_cude / num_points >= tolerance

    @staticmethod
    def intersect_bboxes(pointcloud: np.ndarray, bboxes: List[np.ndarray], tolerance: Optional[float] = 0.1) -> bool:
        """
        Checks if a given pointcloud intersects any of the bounding boxes defines by corner points.

        :param pointcloud: pointcloud of shape (num_points, 3).
        :param bbox: corners of a bounding boxes, as a list of arrays of shape (8, 3)
        :param tolerance: tolerance of intersection.
        :return: bool indicating if the pointcloud intersects any of the bounding boxes.
        """
        for bbox in bboxes:
            if PointcloudPlacement.intersect_bbox(pointcloud=pointcloud, bbox=bbox, tolerance=tolerance):
                return True
        return False
