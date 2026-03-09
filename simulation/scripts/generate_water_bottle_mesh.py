#!/usr/bin/env python3
"""
Generate a realistic plastic water bottle OBJ mesh for MuJoCo simulation.

Creates a surface-of-revolution water bottle with:
  - Flat bottom with slight foot ring
  - Cylindrical body with horizontal ridges
  - Grip waist indent
  - Tapered shoulder
  - Threaded neck with ring
  - Screw cap with lip

Usage:
    cd simulation/scripts
    python generate_water_bottle_mesh.py
"""

import numpy as np
from pathlib import Path


def generate_water_bottle(output_path, n_segments=32):
    """Generate a realistic plastic water bottle mesh as an OBJ file.

    The bottle is centered at the origin (z from -h/2 to h/2).
    Dimensions chosen to fit the WX250s gripper (opens to ~37mm per finger).
    More detailed profile with ridges, grip waist, and thread ring.
    """
    # Profile points (z_local from bottom, radius) in meters
    # Designed to look like a typical plastic water bottle
    profile = [
        (0.000, 0.017),   # bottom edge (slight foot)
        (0.002, 0.019),   # bottom foot ring
        (0.005, 0.017),   # foot indent
        (0.008, 0.021),   # body start (flares out)
        (0.012, 0.022),   # lower body
        # Lower body ridges (horizontal rings typical of plastic bottles)
        (0.022, 0.022),   # below ridge 1
        (0.024, 0.0205),  # ridge 1 (inward)
        (0.026, 0.022),   # above ridge 1
        (0.036, 0.022),   # below ridge 2
        (0.038, 0.0205),  # ridge 2 (inward)
        (0.040, 0.022),   # above ridge 2
        # Grip waist (where you hold the bottle)
        (0.048, 0.022),   # above ridge, entering waist
        (0.053, 0.019),   # waist narrowest point
        (0.058, 0.022),   # exiting waist
        # Upper body
        (0.066, 0.022),   # upper body
        (0.072, 0.0205),  # upper ridge
        (0.074, 0.022),   # above upper ridge
        (0.082, 0.022),   # shoulder start
        # Shoulder taper
        (0.088, 0.021),   # shoulder curve
        (0.094, 0.018),   # shoulder mid
        (0.100, 0.014),   # shoulder narrow
        # Neck
        (0.105, 0.012),   # neck start
        (0.108, 0.013),   # thread ring (wider bump)
        (0.111, 0.012),   # above thread ring
        (0.125, 0.011),   # neck cylinder
        # Cap
        (0.127, 0.0135),  # cap lip (overhangs neck)
        (0.140, 0.0135),  # cap top edge
    ]

    height = profile[-1][0]
    z_offset = -height / 2  # center vertically

    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)

    vertices = []
    faces = []

    # Vertex 1: bottom center
    vertices.append((0.0, 0.0, z_offset))

    # Generate ring vertices for each profile point
    ring_start_indices = []
    for z_local, r in profile:
        z = z_local + z_offset
        ring_start = len(vertices) + 1  # OBJ is 1-indexed
        ring_start_indices.append(ring_start)
        for angle in angles:
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vertices.append((x, y, z))

    # Top center vertex
    top_center_idx = len(vertices) + 1
    vertices.append((0.0, 0.0, height + z_offset))

    # --- Faces ---

    # Bottom cap: fan from center (vertex 1) to first ring
    first_ring = ring_start_indices[0]
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append((1, first_ring + next_i, first_ring + i))

    # Side quads between consecutive rings (two triangles each)
    for ring_idx in range(len(profile) - 1):
        r1 = ring_start_indices[ring_idx]
        r2 = ring_start_indices[ring_idx + 1]
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            v1 = r1 + i
            v2 = r1 + next_i
            v3 = r2 + next_i
            v4 = r2 + i
            faces.append((v1, v2, v3))
            faces.append((v1, v3, v4))

    # Top cap: fan from last ring to top center
    last_ring = ring_start_indices[-1]
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces.append((top_center_idx, last_ring + i, last_ring + next_i))

    # Write OBJ
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# Water bottle mesh for MuJoCo simulation\n")
        f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n")
        f.write(f"# Height: {height:.3f}m, Body radius: 0.020m, Neck radius: 0.011m\n")
        f.write(f"# Centered at origin (z from {z_offset:.3f} to {-z_offset:.3f})\n\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Generated: {output_path}")
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"  Height: {height*100:.1f}cm, Body diameter: 4.4cm, Neck diameter: 2.2cm")
    print(f"  Features: foot ring, body ridges, grip waist, thread ring, cap")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output = script_dir / "../assets/meshes/objects/water_bottle.obj"
    generate_water_bottle(output)
