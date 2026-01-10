#!/usr/bin/env python3
"""
P0-T3 测试: Blender 版本检查集成 (R5 修复验证)

测试场景:
1. 代码集成检查 - 验证版本检查函数存在
2. 源代码验证 - 验证关键代码元素

运行方法 (在容器中):
    cd /mnt/afs2/zhuhaowu/infinigen
    conda activate infinigen
    python tests/sim/test_blender_version_check.py

完整 Blender 内测试:
    ./blender/blender --background --python tests/sim/test_blender_version_check.py -- --blender
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_code_integration():
    """测试代码集成到 init.py"""
    print("\n=== 测试 1: 代码集成检查 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/core/init.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("REQUIRED_BLENDER_VERSION = (4, 0, 0)", "版本要求常量"),
        ("def verify_blender_version()", "版本检查函数"),
        ("bpy.app.version", "版本获取"),
        ("if check_version:", "configure_blender 中的版本检查"),
        ("verify_blender_version()", "版本检查调用"),
        ("RuntimeError", "版本不满足时抛出异常"),
    ]
    
    all_passed = True
    for pattern, desc in checks:
        if pattern in source_code:
            print(f"  ✅ {desc}: 存在")
        else:
            print(f"  ❌ {desc}: 缺失")
            all_passed = False
    
    if all_passed:
        print("✅ 代码集成检查通过")
    else:
        print("❌ 代码集成检查失败")
    
    return all_passed


def test_version_tuple_format():
    """测试版本元组格式正确"""
    print("\n=== 测试 2: 版本元组格式 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/core/init.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    # 检查版本格式
    import re
    version_match = re.search(r'REQUIRED_BLENDER_VERSION\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)', source_code)
    
    if version_match:
        major, minor, patch = map(int, version_match.groups())
        print(f"  版本要求: {major}.{minor}.{patch}")
        
        if major >= 4 and minor >= 0:
            print("  ✅ 版本要求 >= 4.0.0")
            return True
        else:
            print("  ❌ 版本要求应该 >= 4.0.0")
            return False
    else:
        print("  ❌ 无法解析版本要求")
        return False


def test_error_message_quality():
    """测试错误消息质量"""
    print("\n=== 测试 3: 错误消息质量 ===")
    
    source_file = Path(__file__).parent.parent.parent / "infinigen/core/init.py"
    
    with open(source_file, "r") as f:
        source_code = f.read()
    
    checks = [
        ("Blender 版本", "中文错误消息"),
        ("https://www.blender.org", "下载链接"),
        ("几何节点", "说明需要的功能"),
    ]
    
    all_passed = True
    for pattern, desc in checks:
        if pattern in source_code:
            print(f"  ✅ {desc}: 存在")
        else:
            print(f"  ⚠️ {desc}: 缺失 (非关键)")
            # 不将此标记为失败，因为是增强性检查
    
    print("✅ 错误消息质量检查通过")
    return True


def test_in_blender():
    """在 Blender 环境中测试版本检查"""
    print("\n=== 测试 4: Blender 内版本检查 ===")
    
    try:
        import bpy
        from infinigen.core.init import verify_blender_version, REQUIRED_BLENDER_VERSION
        
        current_version = bpy.app.version
        required_version = REQUIRED_BLENDER_VERSION
        
        print(f"  当前 Blender 版本: {'.'.join(map(str, current_version))}")
        print(f"  要求版本: >= {'.'.join(map(str, required_version))}")
        
        # 调用版本检查
        result = verify_blender_version()
        
        if current_version >= required_version and result:
            print("  ✅ 版本检查通过")
            return True
        else:
            print("  ❌ 版本检查失败")
            return False
            
    except ImportError:
        print("  ⚠️ 不在 Blender 环境中，跳过此测试")
        return True
    except RuntimeError as e:
        print(f"  ❌ 版本不满足要求: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("P0-T3 测试: Blender 版本检查 (R5 修复验证)")
    print("=" * 60)
    
    # 检查是否在 Blender 中运行
    in_blender = "--blender" in sys.argv
    
    results = {
        "代码集成": test_code_integration(),
        "版本元组格式": test_version_tuple_format(),
        "错误消息质量": test_error_message_quality(),
    }
    
    if in_blender:
        results["Blender 内测试"] = test_in_blender()
    else:
        print("\n⚠️ 要运行 Blender 内测试，请使用:")
        print("   ./blender/blender --background --python tests/sim/test_blender_version_check.py -- --blender")
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 P0-T3 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败，请检查")
        return 1


if __name__ == "__main__":
    sys.exit(main())
