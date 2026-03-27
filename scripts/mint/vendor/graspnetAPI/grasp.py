"""Minimal Grasp / GraspGroup implementation for AnyGrasp detection wrappers.

This shim intentionally implements only the detection-time surface that the
AnyGrasp SDK and our wrappers rely on. It avoids pulling in the full
graspnetAPI evaluation stack and its heavy DexNet dependencies.
"""

from __future__ import annotations

import copy

import numpy as np

GRASP_ARRAY_LEN = 17


class Grasp:
    def __init__(self, *args):
        if len(args) == 0:
            self.grasp_array = np.array(
                [
                    0.0,
                    0.02,
                    0.02,
                    0.02,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
                dtype=np.float64,
            )
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self.grasp_array = copy.deepcopy(args[0]).astype(np.float64)
        elif len(args) == 7:
            score, width, height, depth, rotation_matrix, translation, object_id = args
            self.grasp_array = np.concatenate(
                [
                    np.array((score, width, height, depth), dtype=np.float64),
                    np.asarray(rotation_matrix, dtype=np.float64).reshape(-1),
                    np.asarray(translation, dtype=np.float64).reshape(-1),
                    np.array((object_id,), dtype=np.float64),
                ]
            )
        else:
            raise ValueError("Unsupported Grasp constructor arguments")

    @property
    def score(self) -> float:
        return float(self.grasp_array[0])

    @property
    def width(self) -> float:
        return float(self.grasp_array[1])

    @property
    def height(self) -> float:
        return float(self.grasp_array[2])

    @property
    def depth(self) -> float:
        return float(self.grasp_array[3])

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.grasp_array[4:13].reshape((3, 3))

    @property
    def translation(self) -> np.ndarray:
        return self.grasp_array[13:16]

    @property
    def object_id(self) -> int:
        return int(self.grasp_array[16])


class GraspGroup:
    def __init__(self, *args):
        if len(args) == 0:
            self.grasp_group_array = np.zeros((0, GRASP_ARRAY_LEN), dtype=np.float64)
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, np.ndarray):
                self.grasp_group_array = np.asarray(arg, dtype=np.float64)
            elif isinstance(arg, str):
                self.grasp_group_array = np.load(arg)
            else:
                raise ValueError("GraspGroup expects a numpy array or path")
        else:
            raise ValueError("Unsupported GraspGroup constructor arguments")

    def __len__(self) -> int:
        return int(len(self.grasp_group_array))

    def __getitem__(self, index):
        if isinstance(index, int):
            return Grasp(self.grasp_group_array[index])
        if isinstance(index, slice):
            return GraspGroup(copy.deepcopy(self.grasp_group_array[index]))
        if isinstance(index, (list, np.ndarray)):
            return GraspGroup(self.grasp_group_array[index])
        raise TypeError(f"Unsupported index type: {type(index)!r}")

    @property
    def scores(self) -> np.ndarray:
        return self.grasp_group_array[:, 0]

    @property
    def widths(self) -> np.ndarray:
        return self.grasp_group_array[:, 1]

    @property
    def heights(self) -> np.ndarray:
        return self.grasp_group_array[:, 2]

    @property
    def depths(self) -> np.ndarray:
        return self.grasp_group_array[:, 3]

    @property
    def rotation_matrices(self) -> np.ndarray:
        return self.grasp_group_array[:, 4:13].reshape((-1, 3, 3))

    @property
    def translations(self) -> np.ndarray:
        return self.grasp_group_array[:, 13:16]

    @property
    def object_ids(self) -> np.ndarray:
        return self.grasp_group_array[:, 16]

    def sort_by_score(self, reverse: bool = False):
        order = np.argsort(self.grasp_group_array[:, 0])
        if not reverse:
            order = order[::-1]
        self.grasp_group_array = self.grasp_group_array[order]
        return self

    def nms(self, translation_thresh: float = 0.03, rotation_thresh: float = np.pi / 6):
        # Detection-only shim: keep API compatibility without pulling in grasp_nms.
        return self

    def to_open3d_geometry_list(self):
        return []
