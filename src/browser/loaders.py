"""Data loading, mesh conversion, and export functions for the model browser."""

import os

import numpy as np

from PySide6.QtCore import QThread, Signal

from pac_parser import PacParser, decompress_type1_pac, material_to_dds_basename
from pam_parser import PamParser, decompress_pam_geometry
from model_types import ParsedModel
from exporters import get_exporter
from exporters.base import ExportOptions
from texture_service import TextureService
from browser.models import CatalogEntry, GpuMesh, SceneMesh, SubmeshInfo


# ── PAC / PAM byte readers ────────────────────────────────────────

def read_pac_bytes(entry) -> bytes:
    """Read and decompress raw PAC bytes from PAZ archive."""
    read_size = entry.comp_size if entry.compressed else entry.orig_size
    with open(entry.paz_file, 'rb') as f:
        f.seek(entry.offset)
        raw = f.read(read_size)
    if entry.compressed and entry.compression_type == 1:
        raw = decompress_type1_pac(raw, entry.orig_size)
    return raw


def read_pam_bytes(entry) -> bytes:
    """Read PAM bytes from PAZ archive and decompress internal geometry if needed."""
    read_size = entry.comp_size if entry.compressed else entry.orig_size
    with open(entry.paz_file, 'rb') as f:
        f.seek(entry.offset)
        raw = f.read(read_size)
    return decompress_pam_geometry(raw)


# ── Mesh conversion ───────────────────────────────────────────────

def _model_to_gpu_mesh(model: ParsedModel) -> GpuMesh:
    """Convert a ParsedModel into a GpuMesh for OpenGL preview."""
    all_pos, all_nor, all_idx = [], [], []
    offset = 0
    for sm in model.submeshes:
        geom = sm.get_geometry(sm.best_lod())
        if geom is None:
            continue
        vb, ib = geom
        all_pos.append(vb.positions)
        all_nor.append(vb.normals)
        all_idx.append(ib.indices.astype(np.uint32) + offset)
        offset += vb.count

    positions = np.concatenate(all_pos)
    normals = np.concatenate(all_nor)
    indices = np.concatenate(all_idx)

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    radius = float(np.linalg.norm(positions - center, axis=1).max())
    if radius < 1e-6:
        radius = 1.0

    return GpuMesh(positions, normals, indices, center, radius)


def _model_to_scene_mesh(model: ParsedModel) -> SceneMesh:
    """Convert a ParsedModel into a SceneMesh with per-submesh info."""
    all_pos, all_nor, all_uv, all_idx = [], [], [], []
    submesh_infos = []
    vertex_offset = 0
    index_offset = 0

    for sm in model.submeshes:
        geom = sm.get_geometry(sm.best_lod())
        if geom is None:
            continue
        vb, ib = geom
        all_pos.append(vb.positions)
        all_nor.append(vb.normals)
        all_uv.append(vb.uvs)
        idx = ib.indices.astype(np.uint32) + vertex_offset
        all_idx.append(idx)

        submesh_infos.append(SubmeshInfo(
            name=sm.name,
            material_name=sm.material_name,
            index_offset=index_offset * 4,  # byte offset (uint32 = 4 bytes)
            index_count=ib.count,
        ))

        vertex_offset += vb.count
        index_offset += ib.count

    positions = np.concatenate(all_pos)
    normals = np.concatenate(all_nor)
    uvs = np.concatenate(all_uv)
    indices = np.concatenate(all_idx)

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    radius = float(np.linalg.norm(positions - center, axis=1).max())
    if radius < 1e-6:
        radius = 1.0

    return SceneMesh(
        positions=positions,
        normals=normals,
        uvs=uvs,
        indices=indices,
        submeshes=submesh_infos,
        center=center,
        radius=radius,
        available_lods=list(model.available_lods),
        current_lod=model.submeshes[0].best_lod() if model.submeshes else 0,
    )


def load_pac_mesh(entry) -> SceneMesh:
    raw = read_pac_bytes(entry)
    parser = PacParser()
    model = parser.parse(raw, lods=[0])
    return _model_to_scene_mesh(model)


def load_pam_mesh(entry) -> SceneMesh:
    raw = read_pam_bytes(entry)
    parser = PamParser()
    model = parser.parse(raw)
    return _model_to_scene_mesh(model)


# ── Export (mesh + materials + DDS textures) ─────────────────────

def export_model_with_textures(entry, output_dir: str,
                               game_dir: str, progress_fn=None,
                               cached_entries=None,
                               apply_dye_colors: bool = False,
                               export_format: str = 'fbx') -> dict:
    """Export PAC model with textures. Format: 'fbx', 'gltf', or 'obj'."""
    pac_data = read_pac_bytes(entry)
    model_name = os.path.splitext(os.path.basename(entry.path))[0]
    model_dir = os.path.join(output_dir, model_name)
    tex_dir = os.path.join(model_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    model = PacParser().parse(pac_data, lods=[0])

    # Extract textures via TextureService
    tex_svc = TextureService(game_dir, cached_entries=cached_entries)
    tex_basenames = [sm.texture_basename for sm in model.submeshes if sm.texture_basename]
    available, extracted = tex_svc.extract_textures(tex_basenames, tex_dir,
                                                     progress_fn=progress_fn)

    # Dye colors via TextureService
    diffuse_overrides = {}
    if apply_dye_colors:
        submesh_pairs = [(sm.name, sm.material_name) for sm in model.submeshes]
        diffuse_overrides = tex_svc.apply_dye_colors(model_name, submesh_pairs,
                                                      tex_dir, progress_fn=progress_fn)

    # Export via selected format
    if progress_fn:
        progress_fn(f"Writing {export_format.upper()}...")

    options = ExportOptions(
        name_hint=model_name,
        texture_rel_dir="textures",
        available_textures=available,
        diffuse_overrides=diffuse_overrides,
    )
    result = get_exporter(export_format).export_to_disk(model, model_dir, options)

    return {
        'output_files': result.output_files,
        'meshes': result.stats['meshes'],
        'vertices': result.stats['vertices'],
        'triangles': result.stats['triangles'],
        'names': result.stats['names'],
        'textures_extracted': extracted,
        'textures_expected': len(set(sm.texture_basename for sm in model.submeshes if sm.texture_basename)),
        'export_dir': model_dir,
        'format': export_format,
    }


def export_pam_with_textures(entry, output_dir: str,
                              game_dir: str, progress_fn=None,
                              cached_entries=None,
                              export_format: str = 'fbx') -> dict:
    """Export PAM model with textures. Format: 'fbx', 'gltf', or 'obj'."""
    pam_data = read_pam_bytes(entry)
    model_name = os.path.splitext(os.path.basename(entry.path))[0]
    model_dir = os.path.join(output_dir, model_name)
    tex_dir = os.path.join(model_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    model = PamParser().parse(pam_data)
    dds_filenames = [sm.texture_basename + '.dds'
                     for sm in model.submeshes if sm.texture_basename]

    # Extract textures via TextureService
    tex_svc = TextureService(game_dir, cached_entries=cached_entries)
    available, extracted = tex_svc.extract_dds_files(dds_filenames, tex_dir,
                                                     progress_fn=progress_fn)

    # Export via selected format
    if progress_fn:
        progress_fn(f"Writing {export_format.upper()}...")

    options = ExportOptions(
        name_hint=model_name,
        texture_rel_dir="textures",
        available_textures=available,
    )
    result = get_exporter(export_format).export_to_disk(model, model_dir, options)

    return {
        'output_files': result.output_files,
        'meshes': result.stats['meshes'],
        'vertices': result.stats['vertices'],
        'triangles': result.stats['triangles'],
        'names': result.stats['names'],
        'textures_extracted': extracted,
        'textures_expected': len(set(dds_filenames)),
        'export_dir': model_dir,
        'format': export_format,
    }


def _get_dest_base(paz_entry, output_dir: str, game_dir: str) -> str:
    """Return output subfolder preserving the exact path from game root
    including the archive folder (e.g. '0009/...' or 'dlc/dlc_01/0009/...').
    """
    paz_dir = os.path.dirname(os.path.abspath(paz_entry.paz_file))
    prefix = os.path.relpath(paz_dir, os.path.abspath(game_dir))
    if prefix and prefix != '.':
        return os.path.join(output_dir, prefix)
    return output_dir


def export_native_with_structure(entry, output_dir: str,
                                 game_dir: str, progress_fn=None,
                                 cached_entries=None) -> dict:
    """Export native model file (.pac/.pam), related .prefab, .pac.xml,
    and .dds textures preserving the game root directory structure (including dlc/...).
    """
    from paz_unpack import extract_entry
    from pac_export import fix_truncated_dds

    output_files = []
    tex_extracted = 0

    if progress_fn:
        progress_fn("Extracting native model file...")

    # 1. Model file (.pac or .pam)
    if entry.path.endswith('.pam'):
        model_data = read_pam_bytes(entry)
        model = PamParser().parse(model_data)
    else:
        model_data = read_pac_bytes(entry)
        model = PacParser().parse(model_data, lods=[0])

    model_dest_base = _get_dest_base(entry, output_dir, game_dir)
    model_rel = entry.path.replace('\\', '/').replace('/', os.sep)
    model_out = os.path.join(model_dest_base, model_rel)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    with open(model_out, 'wb') as f:
        f.write(model_data)
    output_files.append(model_out)

    # 2. Extract textures and prefabs in their exact folder paths
    tex_basenames = set(sm.texture_basename for sm in model.submeshes if sm.texture_basename)
    model_stem = os.path.splitext(os.path.basename(entry.path))[0].lower()

    if cached_entries:
        if progress_fn:
            progress_fn("Extracting native prefabs and textures...")

        for e in cached_entries:
            bn = os.path.basename(e.path).lower()
            p_lower = e.path.lower()
            dest_base = _get_dest_base(e, output_dir, game_dir)

            # Matching prefabs
            if model_stem in bn and p_lower.endswith('.prefab'):
                res = extract_entry(e, dest_base, decrypt_xml=True)
                output_files.append(res['path'])

            # Matching properties / physics (.pac.xml, .meshinfo, .hkx)
            elif model_stem in bn and (p_lower.endswith('.pac.xml') or p_lower.endswith('.meshinfo') or p_lower.endswith('.hkx')):
                res = extract_entry(e, dest_base, decrypt_xml=True)
                output_files.append(res['path'])

            # Matching textures (.dds)
            elif any(bn.startswith(tb.lower()) and p_lower.endswith('.dds') for tb in tex_basenames):
                res = extract_entry(e, dest_base, decrypt_xml=False)
                # Fix truncated DDS headers
                try:
                    with open(res['path'], 'rb') as fp:
                        raw = fp.read()
                    fixed = fix_truncated_dds(raw)
                    if len(fixed) != len(raw):
                        with open(res['path'], 'wb') as fp:
                            fp.write(fixed)
                except Exception:
                    pass
                output_files.append(res['path'])
                tex_extracted += 1

    verts = sum(sm.get_geometry(sm.best_lod())[0].count for sm in model.submeshes if sm.get_geometry(sm.best_lod()))
    tris = sum(sm.get_geometry(sm.best_lod())[1].count // 3 for sm in model.submeshes if sm.get_geometry(sm.best_lod()))

    return {
        'output_files': output_files,
        'meshes': len(model.submeshes),
        'vertices': verts,
        'triangles': tris,
        'names': [sm.name for sm in model.submeshes],
        'textures_extracted': tex_extracted,
        'textures_expected': len(tex_basenames),
        'export_dir': output_dir,
        'format': 'native',
    }


# ── Background workers ────────────────────────────────────────────

class LoadWorker(QThread):
    mesh_ready = Signal(object)
    load_error = Signal(str)

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self._entry = entry

    def run(self):
        try:
            if self._entry.file_type == "pam":
                mesh = load_pam_mesh(self._entry.paz_entry)
            else:
                mesh = load_pac_mesh(self._entry.paz_entry)
            self.mesh_ready.emit(mesh)
        except Exception as e:
            self.load_error.emit(f"{self._entry.filename}: {e}")


class ExportWorker(QThread):
    export_done = Signal(dict)
    export_error = Signal(str)
    progress = Signal(str)

    def __init__(self, entry, output_dir, game_dir, cached_entries=None,
                 apply_dye_colors=False, export_format='fbx', parent=None):
        super().__init__(parent)
        self._entry = entry
        self._output_dir = output_dir
        self._game_dir = game_dir
        self._cached_entries = cached_entries
        self._apply_dye_colors = apply_dye_colors
        self._export_format = export_format

    def run(self):
        try:
            if self._export_format == "native":
                result = export_native_with_structure(
                    self._entry.paz_entry, self._output_dir,
                    self._game_dir, progress_fn=self.progress.emit,
                    cached_entries=self._cached_entries)
            elif self._entry.file_type == "pam":
                result = export_pam_with_textures(
                    self._entry.paz_entry, self._output_dir,
                    self._game_dir, progress_fn=self.progress.emit,
                    cached_entries=self._cached_entries,
                    export_format=self._export_format)
            else:
                result = export_model_with_textures(
                    self._entry.paz_entry, self._output_dir,
                    self._game_dir, progress_fn=self.progress.emit,
                    cached_entries=self._cached_entries,
                    apply_dye_colors=self._apply_dye_colors,
                    export_format=self._export_format)
            self.export_done.emit(result)
        except Exception as e:
            self.export_error.emit(str(e))
