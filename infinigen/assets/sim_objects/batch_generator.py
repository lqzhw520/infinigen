"""
批量盒子生成器框架

Phase 1 Task 6 (P1-T6) 实现:
- 参数空间采样 (均匀、拉丁超立方、网格、Sobol)
- 批量生成管理
- 验证管线集成
- 导出管理
- 生成报告

依赖:
- P1-T1: ModularBoxFactory
- P1-T2: box_geometry_modules
- P1-T3: joint_injector
- P1-T4: material_definitions (Cardboard/Corrugated)
- P1-T5: joint_dynamics
"""

import os
import json
import random
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from datetime import datetime


# ============================================================
# 枚举类型
# ============================================================

class SamplingStrategy(Enum):
    """采样策略枚举"""
    UNIFORM = "uniform"           # 均匀随机采样
    LATIN_HYPERCUBE = "lhs"       # 拉丁超立方采样
    GRID = "grid"                 # 网格采样
    SOBOL = "sobol"               # Sobol 序列采样


# ============================================================
# 数据类
# ============================================================

@dataclass
class GenerationConfig:
    """
    生成配置
    
    定义批量生成的参数和选项
    """
    count: int = 10                                    # 生成数量
    box_types: List[str] = field(default_factory=list)  # 盒型列表
    materials: List[str] = field(default_factory=list)  # 材质列表
    sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    dimension_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    output_dir: str = "generated_boxes"
    seed: Optional[int] = None
    max_retries: int = 3
    parallel: bool = False                             # 是否并行生成
    export_format: str = "urdf"                        # 导出格式
    validate: bool = True                              # 是否验证
    
    def __post_init__(self):
        """设置默认值"""
        if not self.box_types:
            self.box_types = ["TUCK_END", "MAILER", "DRAWER"]
        if not self.materials:
            self.materials = ["cardboard"]
        if not self.dimension_range:
            self.dimension_range = {
                "width": (0.1, 0.5),
                "depth": (0.1, 0.4),
                "height": (0.1, 0.3),
            }
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "count": self.count,
            "box_types": self.box_types,
            "materials": self.materials,
            "sampling_strategy": self.sampling_strategy.name,
            "dimension_range": self.dimension_range,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "max_retries": self.max_retries,
            "parallel": self.parallel,
            "export_format": self.export_format,
            "validate": self.validate,
        }


@dataclass
class GenerationResult:
    """
    生成结果
    
    单个资产的生成结果
    """
    success: bool
    urdf_path: Optional[str] = None
    box_type: str = ""
    material: str = ""
    seed: int = 0
    dimensions: Dict[str, float] = field(default_factory=dict)
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    generation_time: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "success": self.success,
            "urdf_path": self.urdf_path,
            "box_type": self.box_type,
            "material": self.material,
            "seed": self.seed,
            "dimensions": self.dimensions,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "generation_time": self.generation_time,
            "error_message": self.error_message,
        }


@dataclass
class BatchStatistics:
    """
    批量生成统计
    
    跟踪生成过程中的统计信息
    """
    total_attempts: int = 0
    successful: int = 0
    failed_validation: int = 0
    failed_generation: int = 0
    retries_used: int = 0
    total_time: float = 0.0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_material: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful / self.total_attempts
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "failed_validation": self.failed_validation,
            "failed_generation": self.failed_generation,
            "retries_used": self.retries_used,
            "total_time": self.total_time,
            "success_rate": self.success_rate,
            "by_type": self.by_type,
            "by_material": self.by_material,
        }


# ============================================================
# 参数空间采样器
# ============================================================

class ParameterSpaceSampler:
    """
    参数空间采样器
    
    支持多种采样策略:
    - 均匀采样: 简单随机
    - 拉丁超立方采样: 确保参数空间均匀覆盖
    - 网格采样: 规则网格
    - Sobol 序列: 低差异序列
    """
    
    def __init__(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM):
        """
        初始化采样器
        
        Args:
            strategy: 采样策略
        """
        self.strategy = strategy
    
    def sample(self, 
               n: int, 
               parameter_ranges: Dict[str, Tuple[float, float]],
               seed: Optional[int] = None) -> List[Dict[str, float]]:
        """
        采样参数
        
        Args:
            n: 采样数量
            parameter_ranges: 参数范围 {"width": (0.1, 0.5), ...}
            seed: 随机种子
            
        Returns:
            采样结果列表 [{"width": 0.3, "depth": 0.2, ...}, ...]
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if self.strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sample(n, parameter_ranges)
        elif self.strategy == SamplingStrategy.LATIN_HYPERCUBE:
            return self._lhs_sample(n, parameter_ranges)
        elif self.strategy == SamplingStrategy.GRID:
            return self._grid_sample(n, parameter_ranges)
        elif self.strategy == SamplingStrategy.SOBOL:
            return self._sobol_sample(n, parameter_ranges)
        else:
            return self._uniform_sample(n, parameter_ranges)
    
    def _uniform_sample(self, 
                        n: int, 
                        ranges: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """均匀随机采样"""
        samples = []
        for _ in range(n):
            sample = {}
            for key, (low, high) in ranges.items():
                sample[key] = random.uniform(low, high)
            samples.append(sample)
        return samples
    
    def _lhs_sample(self, 
                    n: int, 
                    ranges: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """
        拉丁超立方采样
        
        确保每个维度的边际分布均匀覆盖参数空间
        """
        keys = list(ranges.keys())
        d = len(keys)
        
        # 生成 LHS 矩阵 (n x d)，值在 [0, 1]
        lhs_matrix = np.zeros((n, d))
        for i in range(d):
            # 对每个维度，将 [0, 1] 分成 n 个区间
            intervals = np.arange(n)
            np.random.shuffle(intervals)  # 随机打乱区间顺序
            
            # 在每个区间内随机采样
            for j in range(n):
                interval_start = intervals[j] / n
                interval_end = (intervals[j] + 1) / n
                lhs_matrix[j, i] = random.uniform(interval_start, interval_end)
        
        # 映射到实际参数范围
        samples = []
        for j in range(n):
            sample = {}
            for i, key in enumerate(keys):
                low, high = ranges[key]
                sample[key] = low + lhs_matrix[j, i] * (high - low)
            samples.append(sample)
        
        return samples
    
    def _grid_sample(self, 
                     n: int, 
                     ranges: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """
        网格采样
        
        生成规则的网格点，总数近似 n
        """
        keys = list(ranges.keys())
        d = len(keys)
        
        # 计算每个维度的点数
        points_per_dim = max(2, int(np.ceil(n ** (1.0 / d))))
        
        # 生成每个维度的值
        dim_values = []
        for key in keys:
            low, high = ranges[key]
            dim_values.append(np.linspace(low, high, points_per_dim))
        
        # 生成网格
        samples = []
        grid = np.meshgrid(*dim_values, indexing='ij')
        
        for idx in np.ndindex(*[points_per_dim] * d):
            if len(samples) >= n:
                break
            sample = {}
            for i, key in enumerate(keys):
                sample[key] = float(grid[i][idx])
            samples.append(sample)
        
        # 如果不够，用均匀采样补充
        while len(samples) < n:
            sample = {}
            for key, (low, high) in ranges.items():
                sample[key] = random.uniform(low, high)
            samples.append(sample)
        
        return samples[:n]
    
    def _sobol_sample(self, 
                      n: int, 
                      ranges: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """
        Sobol 序列采样
        
        使用低差异序列提供更均匀的覆盖
        """
        keys = list(ranges.keys())
        d = len(keys)
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True)
            sobol_samples = sampler.random(n)
        except ImportError:
            # 如果没有 scipy，回退到 LHS
            return self._lhs_sample(n, ranges)
        
        # 映射到实际参数范围
        samples = []
        for j in range(n):
            sample = {}
            for i, key in enumerate(keys):
                low, high = ranges[key]
                sample[key] = low + sobol_samples[j, i] * (high - low)
            samples.append(sample)
        
        return samples


# ============================================================
# 验证管线
# ============================================================

class ValidationPipeline:
    """
    验证管线
    
    集成多种验证器，验证生成的资产
    """
    
    def __init__(self):
        """初始化验证管线"""
        self.validators: List[Callable] = []
    
    def add_validator(self, validator: Callable):
        """
        添加验证器
        
        Args:
            validator: 验证函数，签名为 (file_path, params) -> (bool, str)
        """
        self.validators.append(validator)
    
    def add_default_validators(self):
        """添加默认验证器"""
        self.add_validator(self._validate_file_exists)
        self.add_validator(self._validate_dimensions)
        self.add_validator(self._validate_thickness)
    
    def validate(self, 
                 file_path: str, 
                 params: Dict,
                 verbose: bool = True) -> Dict:
        """
        执行验证
        
        Args:
            file_path: 文件路径
            params: 生成参数
            verbose: 是否输出详细信息
            
        Returns:
            验证结果 {"overall_passed": bool, "errors": [...], "warnings": [...]}
        """
        errors = []
        warnings = []
        
        for validator in self.validators:
            try:
                passed, message = validator(file_path, params)
                if not passed:
                    errors.append(message)
                    if verbose:
                        print(f"  ❌ {message}")
            except Exception as e:
                errors.append(f"验证器错误: {e}")
        
        return {
            "overall_passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def _validate_file_exists(self, file_path: str, params: Dict) -> Tuple[bool, str]:
        """验证文件存在"""
        if os.path.exists(file_path):
            return True, "文件存在"
        return False, f"文件不存在: {file_path}"
    
    def _validate_dimensions(self, file_path: str, params: Dict) -> Tuple[bool, str]:
        """验证尺寸合理"""
        dims = params.get("dimensions", {})
        
        for key, val in dims.items():
            if val <= 0:
                return False, f"尺寸 {key} 必须为正数"
            if val > 2.0:
                return False, f"尺寸 {key} 超过最大值 2.0m"
        
        return True, "尺寸有效"
    
    def _validate_thickness(self, file_path: str, params: Dict) -> Tuple[bool, str]:
        """验证壁厚合理"""
        thickness = params.get("thickness", 0.002)
        dims = params.get("dimensions", {})
        
        if not dims:
            return True, "无尺寸信息"
        
        min_dim = min(dims.values())
        
        # 壁厚不应超过最小尺寸的 10%
        if thickness > min_dim * 0.1:
            return False, f"壁厚 ({thickness:.3f}m) 超过最小尺寸 ({min_dim:.3f}m) 的 10%"
        
        return True, "壁厚有效"


# ============================================================
# 导出管理器
# ============================================================

class ExportManager:
    """
    导出管理器
    
    管理生成资产的导出和组织
    """
    
    def __init__(self, base_dir: str = "generated_boxes"):
        """
        初始化导出管理器
        
        Args:
            base_dir: 基础输出目录
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def get_output_path(self, box_type: str, seed: int) -> str:
        """
        获取输出路径
        
        Args:
            box_type: 盒型名称
            seed: 随机种子
            
        Returns:
            输出目录路径
        """
        output_dir = os.path.join(self.base_dir, box_type, f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def save_metadata(self, output_dir: str, params: Dict):
        """
        保存元数据
        
        Args:
            output_dir: 输出目录
            params: 生成参数
        """
        metadata_path = os.path.join(output_dir, "metadata.json")
        
        # 序列化参数
        serializable_params = self._make_serializable(params)
        
        with open(metadata_path, "w") as f:
            json.dump(serializable_params, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.name
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return obj


# ============================================================
# 批量生成器
# ============================================================

class BatchBoxGenerator:
    """
    批量盒子生成器
    
    功能:
    1. 参数空间采样
    2. 批量生成
    3. 自动验证
    4. 失败重试
    5. 生成报告
    """
    
    def __init__(self,
                 box_types: List[str] = None,
                 materials: List[str] = None,
                 output_dir: str = "generated_boxes"):
        """
        初始化批量生成器
        
        Args:
            box_types: 支持的盒型列表
            materials: 支持的材质列表
            output_dir: 输出目录
        """
        self.box_types = box_types or ["TUCK_END", "MAILER", "DRAWER"]
        self.materials = materials or ["cardboard"]
        self.output_dir = output_dir
        
        # 统计信息
        self.stats = BatchStatistics()
        
        # 验证管线
        self.validation_pipeline = ValidationPipeline()
        self.validation_pipeline.add_default_validators()
        
        # 导出管理器
        self.export_manager = ExportManager(output_dir)
    
    def generate_batch(self, config: GenerationConfig) -> List[GenerationResult]:
        """
        批量生成盒子资产
        
        Args:
            config: 生成配置
            
        Returns:
            生成结果列表
        """
        # 重置统计
        self.stats = BatchStatistics()
        
        # 设置输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        self.export_manager = ExportManager(config.output_dir)
        
        # 设置随机种子
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        # 参数采样
        sampler = ParameterSpaceSampler(config.sampling_strategy)
        dimension_samples = sampler.sample(
            config.count, 
            config.dimension_range,
            seed=config.seed
        )
        
        results = []
        
        for i, dim_sample in enumerate(dimension_samples):
            success = False
            retries = 0
            
            while not success and retries <= config.max_retries:
                self.stats.total_attempts += 1
                
                try:
                    # 采样盒型和材质
                    box_type = random.choice(config.box_types)
                    material = random.choice(config.materials)
                    
                    # 生成种子
                    seed = random.randint(0, 2**31 - 1)
                    
                    # 构建完整参数
                    params = {
                        "box_type": box_type,
                        "material": material,
                        "dimensions": dim_sample,
                        "thickness": self._sample_thickness(material, dim_sample),
                        "seed": seed,
                    }
                    
                    # 模拟生成 (实际生成需要 Blender 环境)
                    output_dir = self.export_manager.get_output_path(box_type, seed)
                    urdf_path = os.path.join(output_dir, f"{box_type.lower()}.urdf")
                    
                    # 创建占位 URDF (实际应调用工厂生成)
                    self._create_placeholder_urdf(urdf_path, params)
                    
                    # 保存元数据
                    self.export_manager.save_metadata(output_dir, params)
                    
                    # 验证
                    if config.validate:
                        validation_result = self.validation_pipeline.validate(
                            urdf_path, params, verbose=False
                        )
                        
                        if not validation_result["overall_passed"]:
                            self.stats.failed_validation += 1
                            retries += 1
                            self.stats.retries_used += 1
                            continue
                    
                    # 成功
                    self.stats.successful += 1
                    self.stats.by_type[box_type] = self.stats.by_type.get(box_type, 0) + 1
                    self.stats.by_material[material] = self.stats.by_material.get(material, 0) + 1
                    
                    result = GenerationResult(
                        success=True,
                        urdf_path=urdf_path,
                        box_type=box_type,
                        material=material,
                        seed=seed,
                        dimensions=dim_sample,
                        validation_passed=True,
                    )
                    results.append(result)
                    success = True
                    
                except Exception as e:
                    self.stats.failed_generation += 1
                    retries += 1
                    self.stats.retries_used += 1
                    
                    if retries > config.max_retries:
                        result = GenerationResult(
                            success=False,
                            error_message=str(e),
                        )
                        results.append(result)
        
        # 生成报告
        self._generate_batch_report(config.output_dir, results)
        
        return results
    
    def _sample_thickness(self, material: str, dimensions: Dict) -> float:
        """采样壁厚"""
        min_dim = min(dimensions.values())
        
        # 壁厚范围: 最小尺寸的 0.5% - 5%
        min_thickness = min_dim * 0.005
        max_thickness = min(min_dim * 0.05, 0.005)  # 最大 5mm
        
        return random.uniform(min_thickness, max_thickness)
    
    def _create_placeholder_urdf(self, urdf_path: str, params: Dict):
        """
        创建占位 URDF
        
        注: 实际生成需要在 Blender 环境中调用工厂
        """
        dims = params["dimensions"]
        box_type = params["box_type"]
        
        urdf_content = f'''<?xml version="1.0"?>
<robot name="{box_type.lower()}">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="{dims.get("width", 0.3)} {dims.get("depth", 0.2)} {dims.get("height", 0.15)}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="{dims.get("width", 0.3)} {dims.get("depth", 0.2)} {dims.get("height", 0.15)}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
'''
        
        os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
        with open(urdf_path, "w") as f:
            f.write(urdf_content)
    
    def _generate_batch_report(self, output_dir: str, results: List[GenerationResult]):
        """生成批量报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats.to_dict(),
            "results": [r.to_dict() for r in results],
        }
        
        report_path = os.path.join(output_dir, "batch_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)


# ============================================================
# 便捷函数
# ============================================================

def quick_generate(
    box_type: str = None,
    material: str = "cardboard",
    count: int = 1,
    output_dir: str = "generated_boxes",
    seed: Optional[int] = None
) -> List[GenerationResult]:
    """
    快速生成盒子
    
    Args:
        box_type: 盒型 (None = 随机)
        material: 材质
        count: 数量
        output_dir: 输出目录
        seed: 随机种子
        
    Returns:
        生成结果列表
    """
    box_types = [box_type] if box_type else ["TUCK_END", "MAILER", "DRAWER"]
    
    generator = BatchBoxGenerator(
        box_types=box_types,
        materials=[material],
        output_dir=output_dir,
    )
    
    config = GenerationConfig(
        count=count,
        box_types=box_types,
        materials=[material],
        output_dir=output_dir,
        seed=seed,
    )
    
    return generator.generate_batch(config)


def generate_test_batch(output_dir: str = "test_batch") -> List[GenerationResult]:
    """
    生成测试批次
    
    生成少量样本用于测试
    
    Args:
        output_dir: 输出目录
        
    Returns:
        生成结果列表
    """
    generator = BatchBoxGenerator(
        box_types=["TUCK_END"],
        materials=["cardboard"],
        output_dir=output_dir,
    )
    
    config = GenerationConfig(
        count=5,
        box_types=["TUCK_END"],
        materials=["cardboard"],
        output_dir=output_dir,
        seed=42,
    )
    
    return generator.generate_batch(config)
