"""
Synthetic image-captioning dataset: procedurally generated colored shapes with
matching template captions. Deliberately NOT pure noise -- there's a real, learnable
image -> caption mapping here (color, shape, position are all recoverable from pixels
and stated in the caption), so Phase B can validate whether the full pipeline (frozen
visual encoder -> Q-Pool -> stochastic source -> predictor -> decoder) can learn
*anything* end-to-end before ever touching a real dataset. This is the "known
generative factors" version of the synthetic-distribution validation the project
decided on, one level more concrete than pure embedding-space synthetic mixtures
(test_07 / test_07b), and one level cheaper than a real captioning dataset.
"""
import random
import torch
from torch.utils.data import Dataset

SHAPES = ["circle", "square", "triangle", "cross"]
COLORS = {
    "red": (0.9, 0.15, 0.15),
    "green": (0.15, 0.75, 0.15),
    "blue": (0.15, 0.15, 0.9),
    "yellow": (0.85, 0.85, 0.1),
    "purple": (0.55, 0.15, 0.7),
}
POSITIONS = {
    "the top left": (0.25, 0.25),
    "the top right": (0.75, 0.25),
    "the center": (0.5, 0.5),
    "the bottom left": (0.25, 0.75),
    "the bottom right": (0.75, 0.75),
}


def render_shape(shape: str, color_rgb, pos_frac, size: int = 224, radius_frac: float = 0.16) -> torch.Tensor:
    cx, cy = pos_frac[0] * size, pos_frac[1] * size
    r = radius_frac * size
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")

    if shape == "circle":
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    elif shape == "square":
        mask = (xs - cx).abs().le(r) & (ys - cy).abs().le(r)
    elif shape == "triangle":
        within_height = (ys >= cy - r) & (ys <= cy + r)
        half_width_at_y = (cy + r - ys) * 0.5
        mask = within_height & (xs - cx).abs().le(half_width_at_y)
    elif shape == "cross":
        vbar = (xs - cx).abs().le(r * 0.3) & (ys - cy).abs().le(r)
        hbar = (ys - cy).abs().le(r * 0.3) & (xs - cx).abs().le(r)
        mask = vbar | hbar
    else:
        raise ValueError(f"unknown shape {shape}")

    bg = 0.05 * torch.rand(3, size, size)
    fg = torch.stack([torch.full((size, size), c) for c in color_rgb], dim=0)
    return torch.where(mask.unsqueeze(0).expand(3, -1, -1), fg, bg)


def make_caption(color: str, shape: str, position: str) -> str:
    return f"a {color} {shape} in {position}"


class SyntheticCaptioningDataset(Dataset):
    """Fixed-length, deterministically-seeded so a given index always yields the same
    (image, caption) pair -- reproducible across runs/workers."""

    def __init__(self, length: int = 20000, seed: int = 0, image_size: int = 224):
        self.length = length
        self.image_size = image_size
        rng = random.Random(seed)
        self._factors = [
            (rng.choice(list(COLORS)), rng.choice(SHAPES), rng.choice(list(POSITIONS)))
            for _ in range(length)
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        color, shape, position = self._factors[idx]
        img = render_shape(shape, COLORS[color], POSITIONS[position], size=self.image_size)
        caption = make_caption(color, shape, position)
        return img, caption


def collate_images_captions(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return images, captions
