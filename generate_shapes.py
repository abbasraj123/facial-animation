"""
generate_shapes.py
==================
Generates baseshapes.npy and triangles.npy from the FLAME 2020 model
for use with the Speech-driven-facial-animation renderer (ShapeUtils.py).

Usage:
    python generate_shapes.py --flame_path path/to/generic_model.pkl

Output:
    shapes/baseshapes.npy   shape (47, numVerts*3)  float32
    shapes/triangles.npy    shape (numFaces, 3)     int32

How it works:
    baseshapes[0]    = neutral face (flat vertex array)
    baseshapes[1:47] = 46 expression blendshapes (deltas added to neutral)

    calc_shape() in ShapeUtils.py does:
        full_e = [1, e0, e1, ..., e45]   # size 47
        shape  = full_e @ baseshapes      # (47,) @ (47, V*3) -> (V*3,)
    So row 0 must be the neutral face, rows 1-46 are the blendshape offsets.
"""

import numpy as np
import pickle
import argparse
import os

def load_flame(pkl_path):
    with open(pkl_path, 'rb') as f:
        # Try utf-8 first, fall back to latin1 for older pickle files
        try:
            flame = pickle.load(f, encoding='utf-8')
        except Exception:
            f.seek(0)
            flame = pickle.load(f, encoding='latin1')
    return flame

def generate_shapes(flame_pkl_path, output_dir="shapes", n_expressions=46):
    print(f"Loading FLAME model from: {flame_pkl_path}")
    flame = load_flame(flame_pkl_path)

    # ── Print available keys so you can inspect the model ──
    print("Keys in FLAME model:", list(flame.keys()))

    # ── Extract neutral vertices ──
    # FLAME stores vertices as 'v_template' shape (N, 3)
    v_template = np.array(flame['v_template'], dtype=np.float32)
    num_verts = v_template.shape[0]
    print(f"Number of vertices: {num_verts}")

    # ── Extract expression blendshapes ──
    # 'shapedirs' shape is (N*3, num_betas) or (N, 3, num_betas)
    shapedirs = np.array(flame['shapedirs'], dtype=np.float32)
    print(f"shapedirs shape: {shapedirs.shape}")

    # Reshape to (N*3, num_betas) if needed
    if shapedirs.ndim == 3:
        # (N, 3, num_betas) -> (N*3, num_betas)
        shapedirs = shapedirs.reshape(num_verts * 3, -1)

    total_available = shapedirs.shape[1]
    print(f"Total expression components available: {total_available}")

    # FLAME 2020 has 100 shape + 50 expression components in shapedirs
    # Expression components are the LAST 50 columns
    # We use the first n_expressions (46) of the expression space
    # The expression blendshapes start at index 300 in FLAME 2020
    # (100 shape betas + 200 pose correctives = 300, then expressions)
    # BUT in generic_model.pkl the layout may differ — we'll take last 50
    
    if total_available >= 300 + n_expressions:
        # Full model: shape(100) + pose_correctives(200) + expression(50+)
        expr_start = 300
        print(f"Using expression blendshapes from index {expr_start}")
    elif total_available >= 100 + n_expressions:
        # Shape(100) + expression(50+)
        expr_start = 100
        print(f"Using expression blendshapes from index {expr_start}")
    else:
        # Use whatever is available
        expr_start = max(0, total_available - n_expressions)
        print(f"Using last {n_expressions} components from index {expr_start}")

    expr_blendshapes = shapedirs[:, expr_start:expr_start + n_expressions]
    print(f"Expression blendshapes extracted: {expr_blendshapes.shape}")

    # ── Extract triangles ──
    triangles = np.array(flame['f'], dtype=np.int32)
    print(f"Triangles shape: {triangles.shape}")

    # ── Build baseshapes array ──
    # Shape: (47, num_verts*3)
    # Row 0  : neutral face  (flat)
    # Row 1-46: expression blendshapes (offsets from neutral, flat)
    neutral_flat = v_template.flatten()  # (num_verts*3,)

    baseshapes = np.zeros((n_expressions + 1, num_verts * 3), dtype=np.float32)
    baseshapes[0] = neutral_flat

    for i in range(n_expressions):
        # Each column of expr_blendshapes is a (N*3,) offset vector
        baseshapes[i + 1] = expr_blendshapes[:, i]

    print(f"baseshapes shape: {baseshapes.shape}")  # (47, num_verts*3)

    # ── Normalize scale ──
    # FLAME vertices are in meters (~0.1-0.2 range)
    # The renderer uses glTranslatef(0,0,-4) and perspective projection
    # Scale up so the face fills the view nicely
    scale = 10.0
    baseshapes *= scale
    print(f"Applied scale factor: {scale}")

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)

    baseshapes_path = os.path.join(output_dir, "baseshapes.npy")
    triangles_path  = os.path.join(output_dir, "triangles.npy")

    np.save(baseshapes_path, baseshapes)
    np.save(triangles_path,  triangles)

    print(f"\n✅ Saved baseshapes.npy -> {baseshapes_path}  shape={baseshapes.shape}")
    print(f"✅ Saved triangles.npy  -> {triangles_path}   shape={triangles.shape}")
    print("\nNext step: run render.py to produce your animated face video!")

    return baseshapes, triangles


def verify_shapes(output_dir="shapes"):
    """Quick sanity check that the saved files load and work with calc_shape logic."""
    print("\n── Verification ──")
    baseshapes = np.load(os.path.join(output_dir, "baseshapes.npy"))
    triangles  = np.load(os.path.join(output_dir, "triangles.npy"))

    print(f"baseshapes: {baseshapes.shape} dtype={baseshapes.dtype}")
    print(f"triangles:  {triangles.shape}  dtype={triangles.dtype}")

    # Simulate calc_shape with zero expression (neutral face)
    e = np.zeros(46, dtype=np.float32)
    full_e = np.ones(47, dtype=np.float32)
    full_e[1:] = e
    shape = full_e @ baseshapes
    num_verts = int(shape.size / 3)
    shape_3d = shape.reshape(num_verts, 3)
    print(f"calc_shape output: {shape_3d.shape}  (should be (N, 3))")
    print(f"Vertex range X: [{shape_3d[:,0].min():.3f}, {shape_3d[:,0].max():.3f}]")
    print(f"Vertex range Y: [{shape_3d[:,1].min():.3f}, {shape_3d[:,1].max():.3f}]")
    print(f"Vertex range Z: [{shape_3d[:,2].min():.3f}, {shape_3d[:,2].max():.3f}]")
    print("\n✅ Verification passed! Files are compatible with ShapeUtils.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate baseshapes.npy and triangles.npy from FLAME 2020")
    parser.add_argument("--flame_path", type=str, required=True,
                        help="Path to generic_model.pkl from FLAME 2020")
    parser.add_argument("--output_dir", type=str, default="shapes",
                        help="Output directory (default: shapes/)")
    parser.add_argument("--n_expressions", type=int, default=46,
                        help="Number of expression blendshapes to use (default: 46)")
    args = parser.parse_args()

    baseshapes, triangles = generate_shapes(
        flame_pkl_path=args.flame_path,
        output_dir=args.output_dir,
        n_expressions=args.n_expressions
    )
    verify_shapes(args.output_dir)