"""
Synthetic image-captioning dataset: procedurally generated colored shapes with
matching template captions. Deliberately NOT pure noise -- there's a real, learnable
image -> caption mapping here (color, shape, size, position are all recoverable from
pixels and stated in the caption), so Phase B can validate whether the full pipeline
(frozen visual encoder -> Q-Pool -> stochastic source -> predictor -> decoder) can
learn *anything* end-to-end before ever touching a real dataset.

VOCABULARY SIZE, REVISED after a real training run: an earlier version of this file
used only 5 colors x 4 shapes x 5 positions = 100 unique caption strings. A 3000-step
Kaggle run on that version showed vicreg_t pinned near collapse for most of training
while recon_loss plateaued at ~0.3 -- suspiciously close to ln(100)/16 ~= 0.288, the
entropy floor for an UNCONDITIONAL guess among 100 equally-likely captions. test_11 /
diagnose_recon_signal.py confirmed the decoder wasn't using per-example embedding
information at that vocabulary size (matched vs. shuffled-embedding loss were nearly
identical). This version expands to 8 colors x 5 shapes x 3 sizes x 9 positions = 1080
unique captions specifically so an unconditional shortcut is no longer a viable way to
reach a low loss -- genuine conditioning on the image is now required to beat the (much
higher) new marginal-guessing floor of ln(1080)/16 ~= 0.437.
"""
import random
import torch
from torch.utils.data import Dataset

SHAPES = ["circle", "square", "triangle", "cross", "diamond"]
COLORS = {
    "red": (0.9, 0.15, 0.15),
    "green": (0.15, 0.75, 0.15),
    "blue": (0.15, 0.15, 0.9),
    "yellow": (0.85, 0.85, 0.1),
    "purple": (0.55, 0.15, 0.7),
    "orange": (0.9, 0.5, 0.1),
    "cyan": (0.1, 0.75, 0.8),
    "pink": (0.95, 0.55, 0.75),
}
SIZES = {"small": 0.09, "medium": 0.16, "large": 0.23}
POSITIONS = {
    "the top left": (0.22, 0.22), "the top": (0.5, 0.22), "the top right": (0.78, 0.22),
    "the left": (0.22, 0.5), "the center": (0.5, 0.5), "the right": (0.78, 0.5),
    "the bottom left": (0.22, 0.78), "the bottom": (0.5, 0.78), "the bottom right": (0.78, 0.78),
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
    elif shape == "diamond":
        mask = (xs - cx).abs() + (ys - cy).abs() <= r
    else:
        raise ValueError(f"unknown shape {shape}")

    bg = 0.05 * torch.rand(3, size, size)
    fg = torch.stack([torch.full((size, size), c) for c in color_rgb], dim=0)
    return torch.where(mask.unsqueeze(0).expand(3, -1, -1), fg, bg)


def make_caption(color: str, shape: str, size_name: str, position: str) -> str:
    return f"a {size_name} {color} {shape} in {position}"


class SyntheticCaptioningDataset(Dataset):
    """Fixed-length, deterministically-seeded so a given index always yields the same
    (image, caption) pair -- reproducible across runs/workers. 1080 unique captions
    (8 colors x 5 shapes x 3 sizes x 9 positions) -- see module docstring for why this
    size matters, not just the earlier 100-caption version."""

    def __init__(self, length: int = 20000, seed: int = 0, image_size: int = 224):
        self.length = length
        self.image_size = image_size
        rng = random.Random(seed)
        self._factors = [
            (rng.choice(list(COLORS)), rng.choice(SHAPES), rng.choice(list(SIZES)), rng.choice(list(POSITIONS)))
            for _ in range(length)
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        color, shape, size_name, position = self._factors[idx]
        img = render_shape(shape, COLORS[color], POSITIONS[position], size=self.image_size,
                            radius_frac=SIZES[size_name])
        caption = make_caption(color, shape, size_name, position)
        return img, caption


def collate_images_captions(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return images, captions
