import numpy as np
from splatstream.synthetic import make_scene
from splatstream.compress import (
    CompressionConfig, compress, importance, encode_payload, progressive_prefix
)

def test_pruning_reduces_count():
    s = make_scene(100, seed=1)
    c = compress(s, CompressionConfig("x", 0.5, 16))
    assert c.n == 50

def test_quantization_preserves_shapes_and_range():
    s = make_scene(100, seed=2)
    c = compress(s, CompressionConfig("x", 0.0, 8))
    assert c.means.shape == s.means.shape
    assert c.colors.min() >= 0 and c.colors.max() <= 1
    assert c.opacity.min() >= 0 and c.opacity.max() <= 1
    assert np.all(c.scales > 0)

def test_lower_precision_payload_is_smaller():
    s = make_scene(200, seed=3)
    p32,_ = encode_payload(s, 32)
    s16 = compress(s, CompressionConfig("x",0,16))
    p16,_ = encode_payload(s16,16)
    assert len(p16) < len(p32)

def test_progressive_prefix():
    s = make_scene(100, seed=4)
    p = progressive_prefix(s, 0.25)
    assert p.n == 25
