from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple


def patch_ur5e_xml_without_meshes(src_ur5e: str | Path, dst_ur5e: str | Path) -> None:
    """
    删除视觉 mesh 相关节点，保留关节、执行器、惯性、碰撞体和 TCP site。

    你给的 ur5e.xml 引用了 assets/*.obj。如果本地没有这些网格文件，MuJoCo 会加载失败。
    这个函数用于生成一个运行时 XML，方便先跑算法逻辑。
    """
    src_ur5e = Path(src_ur5e)
    dst_ur5e = Path(dst_ur5e)
    tree = ET.parse(src_ur5e)
    root = tree.getroot()

    for asset in root.findall("asset"):
        for child in list(asset):
            if child.tag == "mesh":
                asset.remove(child)

    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("mesh") is not None:
                parent.remove(child)

    dst_ur5e.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_ur5e, encoding="utf-8", xml_declaration=True)


def prepare_runtime_xml(
    scene_xml: str | Path,
    ur5e_xml: str | Path,
    patch_meshes: bool,
) -> Tuple[Path, tempfile.TemporaryDirectory]:
    """把 scene.xml 和 ur5e.xml 复制到临时目录，确保 include file="ur5e.xml" 可用。"""
    tmp = tempfile.TemporaryDirectory(prefix="tgod_sd_mujoco_")
    tmpdir = Path(tmp.name)

    scene_src = Path(scene_xml)
    ur5e_src = Path(ur5e_xml)
    scene_dst = tmpdir / "scene.xml"
    ur5e_dst = tmpdir / "ur5e.xml"

    scene_tree = ET.parse(scene_src)
    scene_root = scene_tree.getroot()
    for inc in scene_root.findall("include"):
        inc.set("file", "ur5e.xml")
    scene_tree.write(scene_dst, encoding="utf-8", xml_declaration=True)

    if patch_meshes:
        patch_ur5e_xml_without_meshes(ur5e_src, ur5e_dst)
    else:
        shutil.copy2(ur5e_src, ur5e_dst)
        src_assets = ur5e_src.parent / "assets"
        if src_assets.exists():
            shutil.copytree(src_assets, tmpdir / "assets", dirs_exist_ok=True)

    return scene_dst, tmp
