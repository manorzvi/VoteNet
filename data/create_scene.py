import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import trimesh.transformations as trans

from pathlib import Path
from typing import Dict


class Parser(object):
    class ShapesStr(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "shapes_dict", {s.split(":")[0]: int(s.split(":")[1]) for s in values.split(",")})

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", type=int, default=None)

        data = parser.add_argument_group("data")
        data.add_argument("--input-dataset-path", type=Path, required=True)
        data.add_argument("--output-path", type=Path, required=True)
        data.add_argument("--mode", type=str, default="train", choices=("train", "test"))
        data.add_argument("--shapes-str", type=str, action=Parser.ShapesStr, help="chair:2,table:1")
        data.add_argument("--num-scenes", type=int, default=10)

        scale = parser.add_argument_group("scale")
        scale.add_argument("--scale-min", type=float, default=1.0, help="FIXME: CURRENTLY NOT SUPPORTED")
        scale.add_argument("--scale-max", type=float, default=1.0, help="FIXME: CURRENTLY NOT SUPPORTED")

        rotate = parser.add_argument_group("rotate")
        rotate.add_argument("--rotate-x", type=float, default=None)
        rotate.add_argument("--rotate-y", type=float, default=None)
        rotate.add_argument("--rotate-z", type=float, default=None)

        translate = parser.add_argument_group("translate")
        translate.add_argument("--translate-x", type=float, default=None)
        translate.add_argument("--translate-y", type=float, default=None)
        translate.add_argument("--translate-z", type=float, default=0.0)

        opts = parser.parse_args()
        return opts


def set_seed(seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def get_random_files_list(shapes_dict: Dict[str, int]):
    objects_list = []
    for shape, quantity in shapes_dict.items():
        path = opts.dataset_path.joinpath(shape).joinpath(mode)
        files = [f.stem for f in path.iterdir()]
        files_ids = [f.split("_")[1] for f in files]
        sampled_ids = random.sample(files_ids, quantity)
        sampled_files = [path.joinpath(shape + "_" + s) for s in sampled_ids]
        objects_list.extend(sampled_files)
    return objects_list


def get_pointcloud_by_path(path: Path) -> Path:
    path = Path(str(path) + "_points.npy")
    pc = np.load(path)
    pc = pc[:, 0:3]
    return pc


def get_bbox_by_path(path: Path) -> Path:
    path = Path(str(path) + "_bbox.npy")
    bbox = np.load(path)
    return bbox


def get_rotation_matrix(opts: argparse.Namespace):
    rand_rotation = 2 * np.pi * np.random.rand(3)
    alpha, beta, gamma = [x if y is None else 0.0 for x, y in
                          zip(rand_rotation, [opts.rotate_x, opts.rotate_y, opts.rotate_z])]

    origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    Rx = trans.rotation_matrix(alpha, xaxis)
    Ry = trans.rotation_matrix(beta, yaxis)
    Rz = trans.rotation_matrix(gamma, zaxis)
    R = trans.concatenate_matrices(Rx, Ry, Rz)
    return R


def scatter_pointcloud(pc: np.ndarray, bbox: np.ndarray = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker="o")
    if bbox is not None:
        ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], marker="^")
    plt.show()


def get_translation_matrix(opts: argparse.Namespace):
    rand_rotation = np.random.rand(3)
    alpha, beta, gamma = [x if y is None else 0.0 for x, y in
                          zip(rand_rotation, [opts.translate_x, opts.translate_y, opts.translate_z])]

    vector = np.asarray([alpha, beta, gamma])
    unit_vector = vector / np.linalg.norm(vector)

    T = trans.translation_matrix(unit_vector)
    return T


if __name__ == "__main__":

    opts = Parser.parse()
    set_seed(opts.seed)

    mode = "test" if opts.test else "train"

    for i in range(opts.num_scenes):

        objects_paths = get_random_files_list(opts.shapes_dict)

        metadata = {
            "objects": objects_paths,
            "rotations": [],
            "translations": [],
        }

        bbox_list = []
        pointcloud = get_pointcloud_by_path(objects_paths[0])
        bbox_list.append(get_bbox_by_path(objects_paths[0]))

        for object_path in objects_paths[1:]:

            # Load Pointcloud
            curr_pc = get_pointcloud_by_path(object_path)
            curr_bbox = get_bbox_by_path(object_path)

            # Rotate Pointcloud
            R = get_rotation_matrix(opts)[0:3, 0:3]
            metadata["rotations"].append(R)
            curr_pc = np.matmul(R, curr_pc.T).T
            curr_bbox = np.matmul(R, curr_bbox.T).T

            # Get the direction for placing object
            T = get_translation_matrix(opts)
            translation_vector = T[0:3, -1]
            placed = False

            # Try to place the object in direction
            while not hop_place_object(curr_pc, bbox_list):
                curr_pc = curr_pc + translation_vector
                curr_bbox = curr_bbox + translation_vector

            # Add the object to the overall pointcloud
            # and the bbox to the bbox list of all objects.
            pointcloud = np.concatenate((pointcloud, curr_pc), dim=0)
            bbox_list.append(curr_bbox)
            metadata["translations"].append(translation_vector)

        unique_id = f"pointcloud_{str(i).zfill(5)}"
        np.save(opts.output_path.joinpath(unique_id))
        metadata_path = opts.output_path.joinpath(unique_id + "_metadata").with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, indent=4)
