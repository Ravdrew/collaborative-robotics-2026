#!/usr/bin/env python3
"""
Generate texture PNGs for a realistic indoor room scene in MuJoCo.

Creates:
  - tile_floor.png   (light gray/cream tile with grout lines)
  - wall_paint.png   (off-white painted drywall)
  - wood_shelf.png   (darker walnut wood planks)
  - ceiling.png      (white ceiling with subtle variation)

Usage:
    cd simulation/scripts
    uv run python generate_room_textures.py
"""

import numpy as np
from pathlib import Path

try:
    import cv2
    USE_CV2 = True
except ImportError:
    from PIL import Image
    USE_CV2 = False


def save_image(path, img_rgb):
    """Save an RGB numpy array as PNG."""
    if USE_CV2:
        # cv2 expects BGR
        cv2.imwrite(str(path), img_rgb[:, :, ::-1])
    else:
        Image.fromarray(img_rgb).save(str(path))


def add_noise(img, std=5):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_tile_floor(size=512):
    """Generate a light tile/linoleum floor texture with grout lines."""
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Light warm gray/cream tile base (RGB)
    base = np.array([205, 200, 190], dtype=np.float32)

    # Grid of tiles (4x4 tiles in this texture, tiled further by texrepeat)
    n_tiles = 4
    tile_size = size // n_tiles
    grout_width = 3
    grout_color = np.array([140, 135, 125], dtype=np.float32)

    for row in range(n_tiles):
        for col in range(n_tiles):
            # Each tile has slight color/brightness variation
            variation = np.random.uniform(-8, 8, 3)
            tile_color = np.clip(base + variation, 0, 255)

            y0 = row * tile_size
            x0 = col * tile_size
            y1 = y0 + tile_size
            x1 = x0 + tile_size
            img[y0:y1, x0:x1] = tile_color

            # Subtle mottling within each tile (like real ceramic/linoleum)
            for scale in [32, 16]:
                patch_h = tile_size // scale
                patch_w = tile_size // scale
                if patch_h < 1 or patch_w < 1:
                    continue
                mottle = np.random.normal(0, 2.0, (patch_h, patch_w, 3))
                if USE_CV2:
                    mottle_up = cv2.resize(
                        mottle.astype(np.float32),
                        (tile_size, tile_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    mottle_up = np.repeat(
                        np.repeat(mottle, scale, axis=0), scale, axis=1
                    )
                    mottle_up = mottle_up[:tile_size, :tile_size]
                img[y0:y1, x0:x1] += mottle_up[:tile_size, :tile_size]

    # Draw grout lines
    for i in range(1, n_tiles):
        pos = i * tile_size
        # Horizontal grout
        img[pos - grout_width // 2:pos + grout_width // 2 + 1, :] = grout_color
        # Vertical grout
        img[:, pos - grout_width // 2:pos + grout_width // 2 + 1] = grout_color

    # Fine noise
    img = add_noise(np.clip(img, 0, 255).astype(np.uint8), std=3)
    return img


def generate_wall_paint(size=512):
    """Generate an off-white painted drywall texture."""
    # Warm off-white base (RGB)
    base = np.array([225, 222, 215], dtype=np.float32)
    img = np.full((size, size, 3), base, dtype=np.float32)

    # Subtle large-scale color variation (like paint roller texture)
    for scale in [64, 32, 16]:
        small = np.random.normal(0, 1.5, (size // scale, size // scale, 3))
        if USE_CV2:
            upscaled = cv2.resize(
                small.astype(np.float32),
                (size, size),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            # Simple nearest-neighbor upscale
            upscaled = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
            upscaled = upscaled[:size, :size]
        img += upscaled

    # Fine-grain noise (drywall texture)
    img = add_noise(np.clip(img, 0, 255).astype(np.uint8), std=3)
    return img


def generate_wood_shelf(size=512):
    """Generate a darker walnut wood plank texture for shelves."""
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Dark walnut base (RGB)
    base = np.array([120, 70, 35], dtype=np.float32)

    # Vertical planks (shelf boards run lengthwise)
    plank_w = size // 4
    for i in range(0, size, plank_w):
        variation = np.random.uniform(-10, 10, 3)
        color = np.clip(base + variation, 0, 255)
        end = min(i + plank_w, size)
        img[:, i:end] = color

        # Dark gap between planks
        if i > 0:
            gap_color = np.clip(base * 0.5, 0, 255)
            img[:, i:i + 2] = gap_color

    # Vertical grain streaks
    for _ in range(size // 3):
        x = np.random.randint(0, size)
        y_start = np.random.randint(0, size // 2)
        length = np.random.randint(size // 4, size)
        brightness = np.random.uniform(-6, 6)
        y_end = min(y_start + length, size)
        img[y_start:y_end, x] = np.clip(
            img[y_start:y_end, x] + brightness, 0, 255
        )

    img = add_noise(img.astype(np.uint8), std=5)

    if USE_CV2:
        img = cv2.GaussianBlur(img[:, :, ::-1], (3, 7), 0)[:, :, ::-1]

    return img


def generate_ceiling(size=512):
    """Generate a white ceiling texture with very subtle variation."""
    base = np.array([240, 240, 238], dtype=np.float32)
    img = np.full((size, size, 3), base, dtype=np.float32)

    # Very subtle large-scale variation
    small = np.random.normal(0, 1.0, (size // 64, size // 64, 3))
    if USE_CV2:
        upscaled = cv2.resize(
            small.astype(np.float32),
            (size, size),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        upscaled = np.repeat(np.repeat(small, 64, axis=0), 64, axis=1)
        upscaled = upscaled[:size, :size]
    img += upscaled

    img = add_noise(np.clip(img, 0, 255).astype(np.uint8), std=2)
    return img


def main():
    script_dir = Path(__file__).parent
    tex_dir = script_dir / "../assets/textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)  # reproducible textures

    textures = {
        "tile_floor.png": generate_tile_floor,
        "wall_paint.png": generate_wall_paint,
        "wood_shelf.png": generate_wood_shelf,
        "ceiling.png": generate_ceiling,
    }

    for name, gen_fn in textures.items():
        path = tex_dir / name
        img = gen_fn()
        save_image(path, img)
        print(f"  Generated: {path} ({img.shape[0]}x{img.shape[1]})")

    print("All textures generated.")


if __name__ == "__main__":
    main()
