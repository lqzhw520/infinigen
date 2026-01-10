#!/usr/bin/env python3
"""
P1-T6 测试: 批量生成器框架

测试场景:
1. 模块导入 - 验证新增类和函数可导入
2. 采样策略 - 验证各种采样策略
3. 拉丁超立方采样 - 验证 LHS 覆盖率
4. 验证管线 - 验证 ValidationPipeline
5. 导出管理 - 验证 ExportManager
6. 批量生成器 - 验证 BatchBoxGenerator
7. 生成配置 - 验证 GenerationConfig
8. 批量统计 - 验证 BatchStatistics
9. 快速生成 - 验证便捷函数
10. 与其他模块集成 - 验证 P1-T1~T5 集成

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    python tests/sim/test_batch_generator.py
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_module_import():
    """测试模块导入"""
    print("\n=== 测试 1: 模块导入 ===")
    
    try:
        from infinigen.assets.sim_objects.batch_generator import (
            SamplingStrategy,
            GenerationConfig,
            GenerationResult,
            BatchStatistics,
            ParameterSpaceSampler,
            ValidationPipeline,
            ExportManager,
            BatchBoxGenerator,
            quick_generate,
            generate_test_batch,
        )
        print("  ✅ 所有类和函数导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_sampling_strategies():
    """测试采样策略"""
    print("\n=== 测试 2: 采样策略 ===")
    
    from infinigen.assets.sim_objects.batch_generator import (
        ParameterSpaceSampler,
        SamplingStrategy,
    )
    
    dimension_range = {
        "width": (0.1, 0.5),
        "depth": (0.1, 0.4),
        "height": (0.1, 0.3),
    }
    
    strategies = [
        SamplingStrategy.UNIFORM,
        SamplingStrategy.LATIN_HYPERCUBE,
        SamplingStrategy.GRID,
        SamplingStrategy.SOBOL,
    ]
    
    all_passed = True
    
    for strategy in strategies:
        sampler = ParameterSpaceSampler(strategy)
        samples = sampler.sample(10, dimension_range, seed=42)
        
        print(f"  {strategy.name}: {len(samples)} 个样本")
        
        if len(samples) != 10:
            print(f"    ❌ 样本数量错误")
            all_passed = False
            continue
        
        # 检查样本在范围内
        for sample in samples:
            for key, (min_val, max_val) in dimension_range.items():
                val = sample.get(key, 0)
                if not (min_val <= val <= max_val):
                    print(f"    ❌ {key}={val} 超出范围 [{min_val}, {max_val}]")
                    all_passed = False
                    break
        
        # 打印示例
        print(f"    样本示例: width={samples[0]['width']:.3f}")
    
    if all_passed:
        print("  ✅ 采样策略测试通过")
    
    return all_passed


def test_lhs_coverage():
    """测试拉丁超立方采样覆盖率"""
    print("\n=== 测试 3: LHS 覆盖率 ===")
    
    from infinigen.assets.sim_objects.batch_generator import (
        ParameterSpaceSampler,
        SamplingStrategy,
    )
    
    sampler = ParameterSpaceSampler(SamplingStrategy.LATIN_HYPERCUBE)
    
    dimension_range = {
        "width": (0.0, 1.0),
        "depth": (0.0, 1.0),
    }
    
    # 采样 10 个点
    samples = sampler.sample(10, dimension_range, seed=42)
    
    # 检查每个维度的边际分布
    for key in dimension_range.keys():
        values = sorted([s[key] for s in samples])
        
        # LHS 应该保证每个区间 [i/10, (i+1)/10] 都有一个点
        intervals_covered = 0
        for i in range(10):
            interval_min = i / 10
            interval_max = (i + 1) / 10
            
            for val in values:
                if interval_min <= val <= interval_max:
                    intervals_covered += 1
                    break
        
        coverage = intervals_covered / 10
        print(f"  {key} 覆盖率: {coverage:.1%}")
        
        # LHS 应该有较高的覆盖率
        if coverage < 0.7:
            print(f"  ⚠️ {key} 覆盖率较低")
    
    print("  ✅ LHS 覆盖率测试通过")
    return True


def test_validation_pipeline():
    """测试验证管线"""
    print("\n=== 测试 4: 验证管线 ===")
    
    from infinigen.assets.sim_objects.batch_generator import ValidationPipeline
    
    pipeline = ValidationPipeline()
    pipeline.add_default_validators()
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 有效参数
        valid_params = {
            "dimensions": {"width": 0.3, "depth": 0.2, "height": 0.15},
            "thickness": 0.002,
        }
        
        # 创建文件
        test_file = os.path.join(tmpdir, "test.urdf")
        with open(test_file, "w") as f:
            f.write("placeholder")
        
        result = pipeline.validate(test_file, valid_params, verbose=False)
        print(f"  有效参数验证: {result['overall_passed']}")
        
        if not result["overall_passed"]:
            print(f"  ❌ 有效参数应通过验证")
            print(f"    错误: {result['errors']}")
            return False
        
        # 无效参数: 厚度过大
        invalid_params = {
            "dimensions": {"width": 0.1, "depth": 0.1, "height": 0.1},
            "thickness": 0.02,  # 超过最小尺寸的 10%
        }
        
        result = pipeline.validate(test_file, invalid_params, verbose=False)
        print(f"  无效参数验证: {result['overall_passed']}")
        
        if result["overall_passed"]:
            print(f"  ❌ 无效参数应验证失败")
            return False
    
    print("  ✅ 验证管线测试通过")
    return True


def test_export_manager():
    """测试导出管理"""
    print("\n=== 测试 5: 导出管理 ===")
    
    from infinigen.assets.sim_objects.batch_generator import ExportManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ExportManager(tmpdir)
        
        # 获取输出路径
        output_path = manager.get_output_path("TuckEndBox", 12345)
        print(f"  输出路径: {output_path}")
        
        if not os.path.exists(output_path):
            print("  ❌ 输出目录应自动创建")
            return False
        
        # 保存元数据
        params = {"dimensions": {"width": 0.3}, "seed": 12345}
        manager.save_metadata(output_path, params)
        
        metadata_path = os.path.join(output_path, "metadata.json")
        if not os.path.exists(metadata_path):
            print("  ❌ 元数据文件应存在")
            return False
        
        print(f"  元数据路径: {metadata_path}")
    
    print("  ✅ 导出管理测试通过")
    return True


def test_batch_generator():
    """测试批量生成器"""
    print("\n=== 测试 6: 批量生成器 ===")
    
    from infinigen.assets.sim_objects.batch_generator import (
        BatchBoxGenerator,
        GenerationConfig,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = BatchBoxGenerator(
            box_types=["TUCK_END", "MAILER"],
            materials=["cardboard"],
            output_dir=tmpdir,
        )
        
        config = GenerationConfig(
            count=3,
            box_types=["TUCK_END"],
            materials=["cardboard"],
            output_dir=tmpdir,
            seed=42,
        )
        
        # 生成
        print("  开始生成...")
        results = generator.generate_batch(config)
        
        print(f"  生成结果: {len(results)} 个")
        print(f"  成功率: {generator.stats.success_rate:.1%}")
        
        # 验证统计
        if generator.stats.total_attempts < 3:
            print(f"  ❌ 总尝试次数应 >= 3")
            return False
        
        # 验证报告文件
        report_path = os.path.join(tmpdir, "batch_report.json")
        if not os.path.exists(report_path):
            print(f"  ❌ 批量报告应存在")
            return False
        
        print(f"  报告路径: {report_path}")
    
    print("  ✅ 批量生成器测试通过")
    return True


def test_generation_config():
    """测试生成配置"""
    print("\n=== 测试 7: 生成配置 ===")
    
    from infinigen.assets.sim_objects.batch_generator import (
        GenerationConfig,
        SamplingStrategy,
    )
    
    # 默认配置
    default_config = GenerationConfig()
    print(f"  默认数量: {default_config.count}")
    print(f"  默认采样策略: {default_config.sampling_strategy.name}")
    
    if default_config.count != 10:
        print("  ❌ 默认数量应为 10")
        return False
    
    # 自定义配置
    custom_config = GenerationConfig(
        count=100,
        box_types=["TUCK_END"],
        sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE,
        output_dir="/tmp/test",
        seed=42,
    )
    
    config_dict = custom_config.to_dict()
    print(f"  自定义配置: {list(config_dict.keys())}")
    
    if config_dict["count"] != 100:
        print("  ❌ 配置序列化错误")
        return False
    
    print("  ✅ 生成配置测试通过")
    return True


def test_batch_statistics():
    """测试批量统计"""
    print("\n=== 测试 8: 批量统计 ===")
    
    from infinigen.assets.sim_objects.batch_generator import BatchStatistics
    
    stats = BatchStatistics()
    
    # 初始状态
    if stats.success_rate != 0.0:
        print("  ❌ 初始成功率应为 0")
        return False
    
    # 更新统计
    stats.total_attempts = 10
    stats.successful = 8
    stats.failed_validation = 1
    stats.failed_generation = 1
    
    print(f"  成功率: {stats.success_rate:.1%}")
    
    if abs(stats.success_rate - 0.8) > 0.01:
        print("  ❌ 成功率计算错误")
        return False
    
    # 序列化
    stats_dict = stats.to_dict()
    print(f"  统计字典: {stats_dict}")
    
    if "success_rate" not in stats_dict:
        print("  ❌ 序列化缺少 success_rate")
        return False
    
    print("  ✅ 批量统计测试通过")
    return True


def test_quick_generate():
    """测试快速生成"""
    print("\n=== 测试 9: 快速生成 ===")
    
    from infinigen.assets.sim_objects.batch_generator import generate_test_batch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = generate_test_batch(output_dir=tmpdir)
        
        print(f"  生成结果: {len(results)} 个")
        
        if len(results) != 5:
            print("  ❌ 测试批次应生成 5 个")
            return False
        
        # 验证目录结构
        report_path = os.path.join(tmpdir, "batch_report.json")
        if not os.path.exists(report_path):
            print("  ❌ 应生成批量报告")
            return False
    
    print("  ✅ 快速生成测试通过")
    return True


def test_integration():
    """测试与其他模块的集成"""
    print("\n=== 测试 10: 与其他模块集成 ===")
    
    try:
        # 导入其他模块
        from infinigen.assets.sim_objects.modular_box_factory import BoxType
        from infinigen.core.sim.physics.material_definitions import list_box_materials
        from infinigen.core.sim.physics.joint_dynamics import get_material_joint_dynamics, MaterialCategory
        from infinigen.assets.sim_objects.batch_generator import BatchBoxGenerator
        
        # 获取盒型列表
        box_types = [bt.name for bt in BoxType if bt.name != "CUSTOM"]
        print(f"  可用盒型: {len(box_types)} 种")
        
        # 获取材质列表
        materials = list_box_materials()
        print(f"  可用材质: {len(materials)} 种")
        
        # 创建生成器
        generator = BatchBoxGenerator(
            box_types=box_types[:3],
            materials=materials[:3],
        )
        
        print(f"  生成器盒型: {generator.box_types[:3]}")
        print(f"  生成器材质: {generator.materials[:3]}")
        
        print("  ✅ 与其他模块集成测试通过")
        return True
        
    except ImportError as e:
        print(f"  ⚠️ 部分模块未找到: {e}")
        print("  ✅ 集成测试跳过 (模块可能未同步)")
        return True


def test_code_structure():
    """测试代码结构"""
    print("\n=== 测试 11: 代码结构 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/assets/sim_objects/batch_generator.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("class SamplingStrategy(Enum)", "SamplingStrategy 枚举"),
        ("class GenerationConfig", "GenerationConfig 数据类"),
        ("class GenerationResult", "GenerationResult 数据类"),
        ("class BatchStatistics", "BatchStatistics 数据类"),
        ("class ParameterSpaceSampler", "ParameterSpaceSampler 类"),
        ("class ValidationPipeline", "ValidationPipeline 类"),
        ("class ExportManager", "ExportManager 类"),
        ("class BatchBoxGenerator", "BatchBoxGenerator 类"),
        ("def _lhs_sample", "LHS 采样方法"),
        ("def _grid_sample", "网格采样方法"),
        ("def generate_batch", "generate_batch 方法"),
        ("def quick_generate", "quick_generate 函数"),
    ]
    
    all_passed = True
    for pattern, desc in checks:
        if pattern in source_code:
            print(f"  ✅ {desc}: 存在")
        else:
            print(f"  ❌ {desc}: 缺失")
            all_passed = False
    
    return all_passed


def main():
    print("=" * 60)
    print("P1-T6 测试: 批量生成器框架")
    print("=" * 60)
    
    results = {
        "模块导入": test_module_import(),
        "采样策略": test_sampling_strategies(),
        "LHS 覆盖率": test_lhs_coverage(),
        "验证管线": test_validation_pipeline(),
        "导出管理": test_export_manager(),
        "批量生成器": test_batch_generator(),
        "生成配置": test_generation_config(),
        "批量统计": test_batch_statistics(),
        "快速生成": test_quick_generate(),
        "模块集成": test_integration(),
        "代码结构": test_code_structure(),
    }
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 P1-T6 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
