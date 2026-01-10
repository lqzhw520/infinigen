# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import os
import sys

pwd = os.getcwd()
sys.path.append(pwd)

# ============================================================
# Box URDF Generation Project: 继承 conda 环境的 Python 包
# ============================================================
# 
# 问题: Blender 使用自己内置的 Python 解释器，无法访问 conda 环境的包。
# 解决方案: 通过环境变量 INFINIGEN_PYTHONPATH 传递 conda 的 site-packages 路径。
#
# 使用方法:
#   export INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))")
#   python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_blender.py
#
# 或者一行命令:
#   INFINIGEN_PYTHONPATH=$(python -c "import site; print(':'.join(site.getsitepackages()))") \
#   python -m infinigen.launch_blender -s tests/sim/test_tuckendbox_blender.py
#

infinigen_pythonpath = os.environ.get("INFINIGEN_PYTHONPATH", "")
if infinigen_pythonpath:
    for path in infinigen_pythonpath.split(":"):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    print(f"[Infinigen] Added PYTHONPATH: {infinigen_pythonpath[:80]}...")
