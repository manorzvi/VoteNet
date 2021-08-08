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

    @staticmethod
    def _colors(scalar_fn: Optional[np.ndarray] = None) -> np.ndarray:
        if scalar_fn is None:
            return None
        norm = co.Normalize(vmin=scalar_fn.min(), vmax=scalar_fn.max())
        return plt.cm.coolwarm(norm(scalar_fn))

    @staticmethod
    def render_bbox(
        ax: plt.Axes,
        bbox: Optional[np.ndarray] = None,
        min_x: float = -10.0,
        max_x: float = 10.0,
        min_y: float = -10.0,
        max_y: float = 10.0,
    ):
        xx, yy = np.meshgrid(
            np.linspace(start=min_x, stop=max_x, num=100), np.linspace(start=min_y, stop=max_y, num=100)
        )
        # Iterate over 3 possible planes' pairs layouts
        for layout in PointcloudPlacement.BBOX_PLANES_LAYOUT:
            normal, bias = PointcloudPlacement.compute_plane_params(points=bbox[layout[0], :])
            a, b, c = normal
            z = -(a * xx + b * yy + bias) / c
            ax.plot_surface(xx, yy, z, alpha=0.5)

            normal, bias = PointcloudPlacement.compute_plane_params(points=bbox[layout[1], :])
            a, b, c = normal
            z = -(a * xx + b * yy + bias) / c
            ax.plot_surface(xx, yy, z, alpha=0.5)
        ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], marker="^")

    @staticmethod
    def render(
        pointcloud: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        figsize: Optional[Tuple[int, int]] = (10, 10),
        scalar_fn: Optional[np.ndarray] = None,
        path: Optional[Union[Path, str]] = None,
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            pointcloud[:, 0],
            pointcloud[:, 1],
            pointcloud[:, 2],
            marker="o",
            color=PointcloudPlacement._colors(scalar_fn),
        )
        if bbox is not None:
            PointcloudPlacement.render_bbox(
                ax=ax,
                bbox=bbox,
                min_x=2 * pointcloud[:, 0].min(),
                max_x=2 * pointcloud[:, 0].max(),
                min_y=2 * pointcloud[:, 1].min(),
                max_y=2 * pointcloud[:, 1].max(),
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if path is not None:
            plt.savefig(path)
        plt.show()
