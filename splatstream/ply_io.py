from __future__ import annotations
from pathlib import Path
import struct
import numpy as np
from .model import GaussianScene

PLY_TYPES = {
    "char": ("i1", 1),
    "uchar": ("u1", 1),
    "int8": ("i1", 1),
    "uint8": ("u1", 1),
    "short": ("<i2", 2),
    "ushort": ("<u2", 2),
    "int16": ("<i2", 2),
    "uint16": ("<u2", 2),
    "int": ("<i4", 4),
    "uint": ("<u4", 4),
    "int32": ("<i4", 4),
    "uint32": ("<u4", 4),
    "float": ("<f4", 4),
    "float32": ("<f4", 4),
    "double": ("<f8", 8),
    "float64": ("<f8", 8),
}
SH_C0 = 0.28209479177387814

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def read_vertex_table(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if not header_lines or header_lines[0] != "ply":
            raise ValueError("Not a PLY file")

        fmt = None
        vertex_count = None
        in_vertex = False
        props = []

        for line in header_lines[1:]:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("List properties in vertex element are not supported")
                props.append((parts[2], parts[1]))

        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"Unsupported PLY format: {fmt}")
        if vertex_count is None:
            raise ValueError("No vertex element found")

        if fmt == "ascii":
            data = {name: np.empty(vertex_count, np.float64) for name, _ in props}
            for i in range(vertex_count):
                values = f.readline().decode("ascii").split()
                if len(values) < len(props):
                    raise ValueError("Malformed ASCII PLY vertex row")
                for (name, typ), value in zip(props, values):
                    data[name][i] = float(value)
            return {k: v.astype(np.float32) for k,v in data.items()}

        dtype_fields = []
        for name, typ in props:
            if typ not in PLY_TYPES:
                raise ValueError(f"Unsupported property type {typ}")
            dtype_fields.append((name, PLY_TYPES[typ][0]))
        arr = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)
        return {name: np.asarray(arr[name], dtype=np.float32) for name, _ in props}

def graphdeco_to_scene(path: str | Path, max_gaussians: int | None = None) -> GaussianScene:
    data = read_vertex_table(path)
    required = ["x","y","z","scale_0","scale_1","scale_2","opacity","f_dc_0","f_dc_1","f_dc_2"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing GraphDeco properties: {missing}")

    means = np.c_[data["x"], data["y"], data["z"]]
    scales = np.exp(np.c_[data["scale_0"], data["scale_1"], data["scale_2"]])
    dc = np.c_[data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]
    colors = np.clip(0.5 + SH_C0 * dc, 0, 1)
    opacity = _sigmoid(data["opacity"])

    rotation = None
    if all(f"rot_{i}" in data for i in range(4)):
        rotation = np.c_[data["rot_0"],data["rot_1"],data["rot_2"],data["rot_3"]]
        rotation /= np.clip(np.linalg.norm(rotation, axis=1, keepdims=True), 1e-8, None)

    rest_names = sorted(
        [k for k in data if k.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    sh_rest = None
    if rest_names:
        sh_rest = np.stack([data[k] for k in rest_names], axis=1)

    scene = GaussianScene(means, scales, colors, opacity, rotation, sh_rest)
    if max_gaussians is not None and scene.n > max_gaussians:
        # deterministic evenly spaced subsample for portable CPU experiments
        idx = np.linspace(0, scene.n-1, max_gaussians, dtype=np.int64)
        scene = scene.subset(idx)
    return scene

def inspect(path: str | Path) -> dict:
    data = read_vertex_table(path)
    return {
        "vertex_count": int(len(next(iter(data.values()))) if data else 0),
        "properties": list(data.keys()),
        "property_count": len(data),
        "has_dc_sh": all(f"f_dc_{i}" in data for i in range(3)),
        "sh_rest_count": len([k for k in data if k.startswith("f_rest_")]),
        "has_rotation": all(f"rot_{i}" in data for i in range(4)),
        "has_scale": all(f"scale_{i}" in data for i in range(3)),
        "has_opacity": "opacity" in data,
    }
