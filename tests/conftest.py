# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Vineet Bansal

try:
    import bpy  # type: ignore
except ModuleNotFoundError:  # Running outside Blender
    bpy = None  # type: ignore

try:
    import gin  # type: ignore
except ModuleNotFoundError:
    gin = None  # type: ignore
import pytest


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    if gin is not None:
        gin.clear_config()
    if bpy is not None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
