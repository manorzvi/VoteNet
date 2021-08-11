import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Dict, Optional, Tuple


class PointcloudRender(object):
    def __init__(self, figsize: Optional[Tuple[float, float]] = (10, 10)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        self.ax = ax

    @staticmethod
    def _scalar_fn_to_colors(scalar_fn: np.ndarray = None):
        if scalar_fn is None:
            return None
        norm = clr.Normalize(vmin=scalar_fn.min(), vmax=scalar_fn.max())
        return plt.cm.coolwarm(norm(scalar_fn))

    def add_pcd(self, pcd: np.ndarray, marker: Optional[str] = "o", color: Optional[np.ndarray] = None):
        self.ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], marker=marker, color=self._scalar_fn_to_colors(color))

    def add_bbox(self, bbox: np.ndarray, marker: Optional[str] = "^"):
        self.ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], marker=marker)

    def add_plane(
        self,
        points: np.ndarray,
        x_min: float = -10.0,
        x_max: float = 10.0,
        x_num: int = 100,
        y_min: float = -10.0,
        y_max: float = 10.0,
        y_num: int = 100,
        surface_kwargs: Dict = dict(alpha=0.5)
    ):
        if points.shape[0] < 3 or points.shape[1] != 3:
            raise RuntimeError

        v1 = points[2] - points[0]
        v2 = points[1] - points[0]
        normal = np.cross(v1, v2)
        a, b, c = normal
        bias = -1.0 * np.dot(normal, points[0])

        xx, yy = np.meshgrid(
            np.linspace(start=x_min, stop=x_max, num=x_num), np.linspace(start=y_min, stop=y_max, num=y_num)
        )

        z = -1. * (a * xx + b * yy + bias) / c
        self.ax.plot_surface(xx, yy, z, **surface_kwargs)

    def save(self, path: Path):
        plt.savefig(path)

    def render(self):
        plt.show()

    def close(self):
        self.__del__()

    def __del__(self):
        plt.close()
