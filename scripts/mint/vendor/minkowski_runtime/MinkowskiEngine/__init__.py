# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
__version__ = "0.5.4"

import os
import sys
import warnings

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# Force OMP_NUM_THREADS setup
if os.cpu_count() > 16 and "OMP_NUM_THREADS" not in os.environ:
    warnings.warn(
        " ".join(
            [
                "The environment variable `OMP_NUM_THREADS` not set. MinkowskiEngine will automatically set `OMP_NUM_THREADS=16`.",
                "If you want to set `OMP_NUM_THREADS` manually, please export it on the command line before running a python script.",
                "e.g. `export OMP_NUM_THREADS=12; python your_program.py`.",
                "It is recommended to set it below 24.",
            ]
        )
    )
    os.environ["OMP_NUM_THREADS"] = str(16)

# Must be imported first to load all required shared libs
import MinkowskiFunctional
import MinkowskiOps
import torch
from diagnostics import print_diagnostics
from MinkowskiBroadcast import (
    MinkowskiBroadcast,
    MinkowskiBroadcastAddition,
    MinkowskiBroadcastConcatenation,
    MinkowskiBroadcastFunction,
    MinkowskiBroadcastMultiplication,
)
from MinkowskiChannelwiseConvolution import MinkowskiChannelwiseConvolution
from MinkowskiCommon import (
    MinkowskiModuleBase,
    convert_to_int_tensor,
)
from MinkowskiConvolution import (
    MinkowskiConvolution,
    MinkowskiConvolutionFunction,
    MinkowskiConvolutionTranspose,
    MinkowskiConvolutionTransposeFunction,
    MinkowskiGenerativeConvolutionTranspose,
)
from MinkowskiCoordinateManager import (
    CoordinateManager,
    CoordsManager,
    set_gpu_allocator,
    set_memory_manager_backend,
)
from MinkowskiEngineBackend._C import (
    BroadcastMode,
    CoordinateMapKey,
    CoordinateMapType,
    GPUMemoryAllocatorType,
    MinkowskiAlgorithm,
    PoolingMode,
    RegionType,
    cuda_version,
    cudart_version,
    get_gpu_memory_info,
    is_cuda_available,
)
from MinkowskiInterpolation import (
    MinkowskiInterpolation,
    MinkowskiInterpolationFunction,
)
from MinkowskiKernelGenerator import (
    KernelGenerator,
    KernelRegion,
    convert_region_type,
    get_kernel_volume,
)
from MinkowskiNetwork import MinkowskiNetwork
from MinkowskiNonlinearity import (
    MinkowskiAdaptiveLogSoftmaxWithLoss,
    MinkowskiAlphaDropout,
    MinkowskiCELU,
    MinkowskiDropout,
    MinkowskiELU,
    MinkowskiGELU,
    MinkowskiHardshrink,
    MinkowskiHardsigmoid,
    MinkowskiHardswish,
    MinkowskiHardtanh,
    MinkowskiLeakyReLU,
    MinkowskiLogSigmoid,
    MinkowskiLogSoftmax,
    MinkowskiPReLU,
    MinkowskiReLU,
    MinkowskiReLU6,
    MinkowskiRReLU,
    MinkowskiSELU,
    MinkowskiSigmoid,
    MinkowskiSiLU,
    MinkowskiSinusoidal,
    MinkowskiSoftmax,
    MinkowskiSoftmin,
    MinkowskiSoftplus,
    MinkowskiSoftshrink,
    MinkowskiSoftsign,
    MinkowskiTanh,
    MinkowskiTanhshrink,
    MinkowskiThreshold,
)
from MinkowskiNormalization import (
    MinkowskiBatchNorm,
    MinkowskiInstanceNorm,
    MinkowskiInstanceNormFunction,
    MinkowskiStableInstanceNorm,
    MinkowskiSyncBatchNorm,
)
from MinkowskiOps import (
    MinkowskiLinear,
    MinkowskiStackCat,
    MinkowskiStackMean,
    MinkowskiStackSum,
    MinkowskiStackVar,
    MinkowskiToDenseTensor,
    MinkowskiToFeature,
    MinkowskiToSparseTensor,
    cat,
    dense_coordinates,
    mean,
    to_sparse,
    to_sparse_all,
    var,
)
from MinkowskiOps import _sum as sum
from MinkowskiPooling import (
    MinkowskiAvgPooling,
    MinkowskiDirectMaxPoolingFunction,
    MinkowskiGlobalAvgPooling,
    MinkowskiGlobalMaxPooling,
    MinkowskiGlobalPooling,
    MinkowskiGlobalPoolingFunction,
    MinkowskiGlobalSumPooling,
    MinkowskiLocalPoolingFunction,
    MinkowskiLocalPoolingTransposeFunction,
    MinkowskiMaxPooling,
    MinkowskiPoolingTranspose,
    MinkowskiSumPooling,
)
from MinkowskiPruning import MinkowskiPruning, MinkowskiPruningFunction
from MinkowskiSparseTensor import SparseTensor
from MinkowskiTensor import (
    SparseTensorOperationMode,
    SparseTensorQuantizationMode,
    clear_global_coordinate_manager,
    global_coordinate_manager,
    set_global_coordinate_manager,
    set_sparse_tensor_operation_mode,
    sparse_tensor_operation_mode,
)
from MinkowskiTensorField import TensorField
from MinkowskiUnion import MinkowskiUnion, MinkowskiUnionFunction
from sparse_matrix_functions import (
    MinkowskiSPMMAverageFunction,
    MinkowskiSPMMFunction,
    spmm,
)

import MinkowskiEngine.modules as modules
import MinkowskiEngine.utils as utils

if not is_cuda_available():
    warnings.warn(
        " ".join(
            [
                "The MinkowskiEngine was compiled with CPU_ONLY flag.",
                "If you want to compile with CUDA support, make sure `torch.cuda.is_available()` is True when you install MinkowskiEngine.",
            ]
        )
    )
