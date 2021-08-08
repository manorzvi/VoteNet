import argparse
import json
import numpy as np
import random

from pathlib import Path
from pc_placement import PointcloudPlacement
from pc_transforms import PointcloudTransforms
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
        data.add_argument("--num-scenes", type=int, default=1)

        placement = parser.add_argument_group("placement")
        placement.add_argument("--tolerance", type=float, default=0.05)
        placement.add_argument("--step-size", type=float, default=10.0)
        placement.add_argument("--save-figure", action="store_true", default=False)

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


def get_random_files_list(dataset_path: Path, shapes_dict: Dict[str, int], mode: str):
    objects_list = []
    for shape, quantity in shapes_dict.items():
        path = dataset_path.joinpath(shape).joinpath(mode)
        files = [f for f in path.iterdir()]
        sampled_files = random.sample(files, quantity)
        objects_list.extend(sampled_files)
    return objects_list


def get_path_data(path: Path) -> Path:
    with open(path, "r") as f:
        data = json.load(f)

    pcd = np.asarray(data["vertices"])
    bbox = np.asarray(data["oriented_bbox"])
    return pcd, bbox, data["centroid"]


if __name__ == "__main__":

    opts = Parser.parse()
    set_seed(opts.seed)
    opts.output_path.mkdir(parents=True, exist_ok=True)

    transforms = PointcloudTransforms(opts=opts, seed=opts.seed)

    for i in range(opts.num_scenes):

        objects_paths = get_random_files_list(
            dataset_path=opts.input_dataset_path, shapes_dict=opts.shapes_dict, mode=opts.mode
        )

        pointclouds, bboxes = [], []
        scene_data = []
        for object_path in objects_paths:

            # Load Pointcloud
            pcd, bbox, _ = get_path_data(object_path)

            # Rotate Pointcloud
            pcd, bbox = transforms.apply_rotation([pcd, bbox])

            # Get the translation vector
            translation_vector = transforms.get_translation_vector(opts.step_size)

            # Try to place the object in direction
            while PointcloudPlacement.intersect_bboxes(pcd, bboxes, tolerance=opts.tolerance):
                pcd += translation_vector
                bbox += translation_vector

            pointclouds.append(pcd)
            bboxes.append(bbox)
            scene_data.append({"object": str(object_path), "pointcloud": pcd.tolist(), "oriented_bbox": bbox.tolist()})

        data_path = opts.output_path.joinpath(str(i).zfill(5))
        pdf_path = data_path.with_suffix(".pdf") if opts.save_figure else None

        with open(data_path.with_suffix(".json"), "w") as f:
            json.dump(scene_data, f, indent=4)
