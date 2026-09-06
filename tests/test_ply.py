from pathlib import Path
from splatstream.ply_io import read_vertex_table, graphdeco_to_scene

def test_ascii_graphdeco_ply(tmp_path: Path):
    p = tmp_path/"tiny.ply"
    p.write_text("""ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
0 0 3 0 0 0 1 -2 -2 -2 1 0 0 0
1 0 4 .1 .2 .3 0 -2.2 -2.2 -2.2 1 0 0 0
""")
    d = read_vertex_table(p)
    assert len(d["x"]) == 2
    scene = graphdeco_to_scene(p)
    assert scene.n == 2
    assert scene.scales.min() > 0
