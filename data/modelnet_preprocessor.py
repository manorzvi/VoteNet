import argparse
import meshio
import numpy as np
import os
import shutil
import sys
import trimesh
import zipfile

from loguru import logger
from pathlib import Path


class SuppressPrints(object):
    def __init__(self, stdout: bool = True, stderr: bool = True):
        self._out = stdout
        self._err = stderr

    def __enter__(self):
        if self._out:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self._err:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._out:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self._err:
            sys.stderr.close()
            sys.stderr = self._original_stderr


def sample_mesh_to_num_points(mesh_data: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """
    Sample num_points from mesh and compute the normals at the points.
    :param mesh_data: trimesh object with a shape.
    :param num_points: number of sampling points.
    :return: sampled points as numpy array and the normals at the points.
                The array is of size (num_points,6).
    """
    with SuppressPrints():
        points, faces = trimesh.sample.sample_surface_even(mesh_data, num_points)
        if points.shape[0] < num_points:
            points, faces = trimesh.sample.sample_surface(mesh_data, num_points)
    normals = mesh_data.face_normals[faces]

    samples = np.concatenate((points, normals), axis=1)
    return samples


def process_dir(source: str, target: str, num_points: int) -> None:
    """
    Process files of a single dir.
     1. read file
     2. sample num_points of the shape.
     3. rotate for later convenient.
    :param source: source dir.
    :param target: target dir.
    :param num_points: number of points to sample.
    """
    for file in source.iterdir():

        try:
            mesh_data = meshio.read(file)
        except meshio._exceptions.ReadError:
            logger.warning(f"Error reading `{file.name}`. Skipping...")
            continue

        trimesh_data = trimesh.Trimesh(vertices=mesh_data.points, faces=mesh_data.cells_dict["triangle"])

        # To put the centroid in the origin
        principal_inertia_transform = trimesh_data.principal_inertia_transform
        transformed_mesh = trimesh_data.apply_transform(principal_inertia_transform)

        sampled_points = sample_mesh_to_num_points(mesh_data=transformed_mesh, num_points=num_points)

        np.save(target.joinpath(file.stem + "_bbox"), transformed_mesh.bounding_box.vertices)
        np.save(target.joinpath(file.stem + "_centroid"), transformed_mesh.centroid)
        np.save(target.joinpath(file.stem + "_points"), sampled_points)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-dataset-path", type=Path, required=True, help="Path to a zipped model-net dataset")
    parser.add_argument("--num-points", type=int, default=1024, help="#points (sampled) for each mesh")
    parser.add_argument("--force", action="store_true", help="Set to force pre-proceesing")
    opts = parser.parse_args()

    ds_path = opts.zip_dataset_path.expanduser()
    if not ds_path.exists() or not zipfile.is_zipfile(ds_path):
        raise ValueError(f"{ds_path} does not exist or not a zip file")

    with zipfile.ZipFile(ds_path, "r") as zip_ref:
        ds_path_unzip = ds_path.parent.joinpath("__unzip__")
        if not ds_path_unzip.exists():
            zip_ref.extractall(ds_path_unzip)
        ds_path_unzip_ = ds_path_unzip.joinpath(ds_path.stem)

    processed_ds_path = ds_path.parent.joinpath(f"{ds_path.stem}_{opts.num_points}")
    if opts.force and opts.processed_ds_path.exists():
        shutil.rmtree(processed_ds_path, ignore_errors=True)
    processed_ds_path.mkdir(parents=True, exist_ok=True)

    for file in ds_path_unzip_.iterdir():

        if not file.is_dir():
            continue

        logger.info(f"Processing `{file.name}`")
        for mode in ("train", "test"):
            processed_path = processed_ds_path.joinpath(file.name).joinpath(mode)
            processed_path.mkdir(parents=True, exist_ok=True)
            process_dir(file.joinpath(mode), processed_path, opts.num_points)

    shutil.rmtree(ds_path_unzip)
