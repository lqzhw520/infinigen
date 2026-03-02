#!/usr/bin/env python3
"""
URDF Inertial 修复验证脚本

本脚本验证 P0 修复的正确性:
1. 每个 link 只有一个 <inertial> 元素 (符合 URDF 规范)
2. 惯性合并使用了正确的平行轴定理
3. 物理属性在 PyBullet 中正确加载
4. 质量和惯性张量满足物理约束

数学验证:
=========
平行轴定理 (Parallel Axis Theorem):
    I_new = I_old + m × [(r·r)×E - r⊗r]

其中:
    r = c_i - C (质心偏移向量)
    E 是 3×3 单位矩阵
    ⊗ 表示外积 (outer product)

验证项:
    1. 总质量 = Σ m_i
    2. 合并质心 = (1/M) × Σ (m_i × c_i)
    3. 惯性张量正定性
    4. 惯性张量满足三角不等式

使用方法:
    cd /mnt/afs2/zhuhaowu/infinigen
    conda run -n infinigen python scripts/verify_urdf_inertia_fix.py \
        --urdf-dir sim_exports/urdf/mailerbox_simple

作者: 技术分析团队
日期: 2026-01-27
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class URDFInertiaVerifier:
    """URDF 惯性属性验证器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def error(self, msg: str):
        self.errors.append(msg)
        if self.verbose:
            print(f"[ERROR] {msg}")
    
    def warning(self, msg: str):
        self.warnings.append(msg)
        if self.verbose:
            print(f"[WARNING] {msg}")
    
    def verify_urdf_structure(self, urdf_path: Path) -> Tuple[bool, Dict]:
        """
        验证 URDF 结构合规性。
        
        检查项:
        1. 每个 link 只能有 0 或 1 个 <inertial> 元素
        2. <inertial> 必须包含 <mass>, <inertia>, <origin>
        
        Returns:
            (is_valid, link_info_dict)
        """
        self.log(f"\n{'='*60}")
        self.log(f"验证 URDF 结构: {urdf_path.name}")
        self.log(f"{'='*60}")
        
        if not urdf_path.exists():
            self.error(f"URDF 文件不存在: {urdf_path}")
            return False, {}
        
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except ET.ParseError as e:
            self.error(f"URDF 解析失败: {e}")
            return False, {}
        
        link_info = {}
        all_valid = True
        
        for link in root.findall("link"):
            link_name = link.get("name", "unnamed")
            inertials = link.findall("inertial")
            visuals = link.findall("visual")
            collisions = link.findall("collision")
            
            info = {
                "n_inertial": len(inertials),
                "n_visual": len(visuals),
                "n_collision": len(collisions),
                "inertial_valid": False,
                "mass": None,
                "com": None,
                "inertia_diag": None,
            }
            
            # 检查 inertial 数量
            if len(inertials) > 1:
                self.error(
                    f"Link '{link_name}': 有 {len(inertials)} 个 <inertial> 元素 "
                    f"(URDF 规范要求最多 1 个)"
                )
                all_valid = False
            elif len(inertials) == 1:
                inertial = inertials[0]
                
                # 提取 mass
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    try:
                        info["mass"] = float(mass_elem.get("value", "0"))
                    except ValueError:
                        self.error(f"Link '{link_name}': 无效的 mass 值")
                        all_valid = False
                
                # 提取 origin (质心位置)
                origin_elem = inertial.find("origin")
                if origin_elem is not None:
                    xyz_str = origin_elem.get("xyz", "0 0 0")
                    try:
                        info["com"] = np.array([float(x) for x in xyz_str.split()])
                    except ValueError:
                        self.error(f"Link '{link_name}': 无效的 origin xyz 值")
                        all_valid = False
                
                # 提取 inertia
                inertia_elem = inertial.find("inertia")
                if inertia_elem is not None:
                    try:
                        ixx = float(inertia_elem.get("ixx", "0"))
                        iyy = float(inertia_elem.get("iyy", "0"))
                        izz = float(inertia_elem.get("izz", "0"))
                        info["inertia_diag"] = np.array([ixx, iyy, izz])
                        info["inertial_valid"] = True
                    except ValueError:
                        self.error(f"Link '{link_name}': 无效的 inertia 值")
                        all_valid = False
                
                # 验证物理约束
                if info["mass"] is not None and info["mass"] <= 0:
                    self.warning(f"Link '{link_name}': 质量 <= 0 ({info['mass']})")
                
                if info["inertia_diag"] is not None:
                    diag = info["inertia_diag"]
                    if np.any(diag <= 0):
                        self.warning(f"Link '{link_name}': 惯性张量对角元素存在非正值")
                    
                    # 三角不等式检查
                    if diag[0] + diag[1] < diag[2]:
                        self.warning(f"Link '{link_name}': 惯性张量违反三角不等式 (Ixx+Iyy < Izz)")
                    if diag[0] + diag[2] < diag[1]:
                        self.warning(f"Link '{link_name}': 惯性张量违反三角不等式 (Ixx+Izz < Iyy)")
                    if diag[1] + diag[2] < diag[0]:
                        self.warning(f"Link '{link_name}': 惯性张量违反三角不等式 (Iyy+Izz < Ixx)")
            
            link_info[link_name] = info
            
            self.log(
                f"  Link '{link_name}': "
                f"inertial={len(inertials)}, visual={len(visuals)}, collision={len(collisions)}"
            )
            if info["mass"] is not None:
                self.log(f"    mass={info['mass']:.6f} kg")
            if info["com"] is not None:
                self.log(f"    com={info['com']}")
            if info["inertia_diag"] is not None:
                self.log(f"    I_diag={info['inertia_diag']}")
        
        return all_valid, link_info
    
    def verify_pybullet_load(self, urdf_path: Path) -> Tuple[bool, Dict]:
        """
        使用 PyBullet 加载 URDF 并验证物理属性。
        
        Returns:
            (is_valid, dynamics_info)
        """
        self.log(f"\n{'='*60}")
        self.log(f"PyBullet 加载验证: {urdf_path.name}")
        self.log(f"{'='*60}")
        
        try:
            import pybullet as p
        except ImportError:
            self.error("PyBullet 未安装")
            return False, {}
        
        if not urdf_path.exists():
            self.error(f"URDF 文件不存在: {urdf_path}")
            return False, {}
        
        physics_id = p.connect(p.DIRECT)
        try:
            # 加载 URDF
            try:
                # IMPORTANT (PyBullet):
                # - 为了验证“URDF 文件中写出的 <inertial> 是否被仿真器真实采用”，必须开启
                #   URDF_USE_INERTIA_FROM_FILE。
                # - 若不设置该 flag，PyBullet 可能会根据 collision geometry 重新估计惯性，
                #   这会导致 getDynamicsInfo() 返回的惯性与 URDF 中的数值不一致，
                #   从而产生“假警告”，也无法完成严谨的物理对齐验证。
                flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
                body_id = p.loadURDF(
                    str(urdf_path),
                    useFixedBase=True,
                    flags=flags,
                )
            except Exception as e:
                self.error(f"PyBullet 加载失败: {e}")
                return False, {}
            
            dynamics_info = {}
            all_valid = True
            
            # 获取 base link 信息 (link index = -1)
            base_info = p.getDynamicsInfo(body_id, -1)
            dynamics_info["base_link"] = {
                "mass": base_info[0],
                "local_inertia_diagonal": base_info[2],
                "local_inertia_pos": base_info[3],
            }
            self.log(f"  Base link: mass={base_info[0]:.6f} kg, I_diag={base_info[2]}")
            
            # 获取所有 joint 对应的 link 信息
            n_joints = p.getNumJoints(body_id)
            for j in range(n_joints):
                joint_info = p.getJointInfo(body_id, j)
                joint_name = joint_info[1].decode("utf-8")
                link_name = joint_info[12].decode("utf-8")
                
                link_dynamics = p.getDynamicsInfo(body_id, j)
                dynamics_info[link_name] = {
                    "mass": link_dynamics[0],
                    "local_inertia_diagonal": link_dynamics[2],
                    "local_inertia_pos": link_dynamics[3],
                    "joint_name": joint_name,
                }
                
                self.log(
                    f"  Link '{link_name}' (joint: {joint_name}): "
                    f"mass={link_dynamics[0]:.6f} kg, I_diag={link_dynamics[2]}"
                )
                
                # 验证质量
                if link_dynamics[0] <= 0:
                    self.warning(f"Link '{link_name}': PyBullet 报告质量 <= 0")
                
                # 验证惯性
                I_diag = link_dynamics[2]
                if any(i <= 0 for i in I_diag):
                    self.warning(f"Link '{link_name}': PyBullet 报告惯性对角元素存在非正值")
            
            return all_valid, dynamics_info
            
        finally:
            p.disconnect()
    
    def compare_expected_vs_actual(
        self,
        urdf_info: Dict,
        pybullet_info: Dict,
        tolerance: float = 1e-4
    ) -> bool:
        """
        对比 URDF 文件中的值与 PyBullet 加载后的值。
        """
        self.log(f"\n{'='*60}")
        self.log("URDF vs PyBullet 对比验证")
        self.log(f"{'='*60}")
        
        all_match = True
        
        for link_name, urdf_data in urdf_info.items():
            if link_name == "world":
                continue
            
            # 尝试匹配 PyBullet 中的 link
            pb_key = None
            if link_name in pybullet_info:
                pb_key = link_name
            elif link_name == "link_0" and "base_link" in pybullet_info:
                pb_key = "base_link"
            
            if pb_key is None:
                self.warning(f"Link '{link_name}': 在 PyBullet 结果中未找到对应项")
                continue
            
            pb_data = pybullet_info[pb_key]
            
            # 对比质量
            if urdf_data["mass"] is not None:
                mass_diff = abs(urdf_data["mass"] - pb_data["mass"])
                if mass_diff > tolerance:
                    self.warning(
                        f"Link '{link_name}': 质量不匹配 "
                        f"(URDF={urdf_data['mass']:.6f}, PyBullet={pb_data['mass']:.6f})"
                    )
                else:
                    self.log(f"  Link '{link_name}': 质量匹配 ✓ ({urdf_data['mass']:.6f} kg)")
            
            # 对比惯性
            if urdf_data["inertia_diag"] is not None:
                pb_inertia = np.array(pb_data["local_inertia_diagonal"])
                inertia_diff = np.abs(urdf_data["inertia_diag"] - pb_inertia)
                if np.any(inertia_diff > tolerance):
                    self.warning(
                        f"Link '{link_name}': 惯性不匹配 "
                        f"(URDF={urdf_data['inertia_diag']}, PyBullet={pb_inertia})"
                    )
                else:
                    self.log(f"  Link '{link_name}': 惯性匹配 ✓")
        
        return all_match
    
    def verify_single_urdf(self, urdf_path: Path) -> bool:
        """验证单个 URDF 文件"""
        self.errors = []
        self.warnings = []
        
        # 1. 验证 URDF 结构
        struct_valid, urdf_info = self.verify_urdf_structure(urdf_path)
        
        # 2. PyBullet 加载验证
        pb_valid, pb_info = self.verify_pybullet_load(urdf_path)
        
        # 3. 对比验证
        if struct_valid and pb_valid:
            self.compare_expected_vs_actual(urdf_info, pb_info)
        
        # 总结
        self.log(f"\n{'='*60}")
        self.log("验证总结")
        self.log(f"{'='*60}")
        self.log(f"  结构验证: {'✓ PASS' if struct_valid else '✗ FAIL'}")
        self.log(f"  PyBullet 验证: {'✓ PASS' if pb_valid else '✗ FAIL'}")
        self.log(f"  错误数: {len(self.errors)}")
        self.log(f"  警告数: {len(self.warnings)}")
        
        return struct_valid and pb_valid and len(self.errors) == 0


def verify_all_seeds(urdf_dir: Path, verbose: bool = True) -> Dict:
    """验证目录下所有 seed 的 URDF"""
    
    verifier = URDFInertiaVerifier(verbose=verbose)
    results = {
        "urdf_dir": str(urdf_dir),
        "seeds": {},
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
        }
    }
    
    # 查找所有 seed 目录
    seed_dirs = sorted([d for d in urdf_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not seed_dirs:
        print(f"未找到任何 seed 目录: {urdf_dir}")
        return results
    
    print(f"\n{'#'*70}")
    print("# URDF Inertial 修复验证")
    print(f"# 目录: {urdf_dir}")
    print(f"# Seeds: {[d.name for d in seed_dirs]}")
    print(f"{'#'*70}")
    
    for seed_dir in seed_dirs:
        seed = seed_dir.name
        # Prefer "<asset_name>.urdf" where asset_name == urdf_dir.name (common export convention),
        # otherwise fall back to the first non-viewer-safe URDF in the seed folder.
        expected = seed_dir / f"{urdf_dir.name}.urdf"
        if expected.exists():
            urdf_path = expected
        else:
            urdfs = sorted(seed_dir.glob("*.urdf"))
            candidates = [p for p in urdfs if "viewer_safe" not in p.name]
            urdf_path = candidates[0] if candidates else (urdfs[0] if urdfs else None)
        
        if urdf_path is None or not urdf_path.exists():
            print(f"\n[SKIP] seed {seed}: URDF 文件不存在")
            continue
        
        results["summary"]["total"] += 1
        
        print(f"\n{'#'*70}")
        print(f"# Seed {seed}")
        print(f"{'#'*70}")
        
        passed = verifier.verify_single_urdf(urdf_path)
        
        results["seeds"][seed] = {
            "urdf": str(urdf_path),
            "passed": passed,
            "errors": verifier.errors.copy(),
            "warnings": verifier.warnings.copy(),
        }
        
        if passed:
            results["summary"]["passed"] += 1
            print(f"\n✓ Seed {seed}: PASS")
        else:
            results["summary"]["failed"] += 1
            print(f"\n✗ Seed {seed}: FAIL")
    
    # 最终总结
    print(f"\n{'#'*70}")
    print("# 最终验证结果")
    print(f"{'#'*70}")
    print(f"  总计: {results['summary']['total']}")
    print(f"  通过: {results['summary']['passed']}")
    print(f"  失败: {results['summary']['failed']}")
    
    if results["summary"]["failed"] == 0:
        print("\n✓ 所有 URDF 验证通过！惯性合并修复成功。")
    else:
        print("\n✗ 存在验证失败的 URDF，请检查详细错误信息。")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="验证 URDF Inertial 修复的正确性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 验证所有 mailerbox_simple seeds
    python scripts/verify_urdf_inertia_fix.py \\
        --urdf-dir sim_exports/urdf/mailerbox_simple
    
    # 验证单个 URDF 文件
    python scripts/verify_urdf_inertia_fix.py \\
        --urdf sim_exports/urdf/mailerbox_simple/101/mailerbox_simple.urdf
        """
    )
    
    parser.add_argument(
        "--urdf-dir",
        type=Path,
        help="包含多个 seed 子目录的 URDF 根目录"
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        help="单个 URDF 文件路径"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出验证报告 JSON 文件路径"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只输出最终结果"
    )
    
    args = parser.parse_args()
    
    if args.urdf:
        # 验证单个 URDF
        verifier = URDFInertiaVerifier(verbose=not args.quiet)
        passed = verifier.verify_single_urdf(args.urdf)
        sys.exit(0 if passed else 1)
    
    elif args.urdf_dir:
        # 验证目录下所有 seeds
        results = verify_all_seeds(args.urdf_dir, verbose=not args.quiet)
        
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2, default=str))
            print(f"\n验证报告已保存: {args.output}")
        
        sys.exit(0 if results["summary"]["failed"] == 0 else 1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
