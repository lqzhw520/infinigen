# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import os
import sys

pwd = os.getcwd()

# ============================================================
# Box URDF Generation Project: 继承 conda 环境的 Python 包
# ============================================================
#
# 目标:
# - 优先使用当前 repo 的源码 (pwd) 以便开发调试
# - 同时让 Blender Python 能访问 conda 环境的依赖 (INFINIGEN_PYTHONPATH)
#
# 关键点:
# - conda 里的 numpy/opencv 等二进制包可能与 Blender 自带 numpy 不兼容
# - 因此需要让 conda 的 site-packages 在 sys.path 中优先于 Blender 自带包，
#   以避免 “cv2 + numpy>=2” 的二进制不兼容报错。

# 1) 先把 repo 根目录放到最前面，确保 import infinigen 指向当前代码
if pwd in sys.path:
    sys.path.remove(pwd)
sys.path.insert(0, pwd)

# 2) 再把 conda site-packages 放到 pwd 之后（仍然优先于 Blender 自带包）
infinigen_pythonpath = os.environ.get("INFINIGEN_PYTHONPATH", "")
if infinigen_pythonpath:
    # 保持原顺序插入到 index=1
    paths = [p for p in infinigen_pythonpath.split(":") if p]
    for p in reversed(paths):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(1, p)
    print(f"[Infinigen] Added PYTHONPATH: {infinigen_pythonpath[:80]}...")
