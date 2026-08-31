"""
Tests for real_captioning_data.py, using a LOCAL stand-in matching
nlphuji/flickr30k's confirmed schema ({'image': PIL.Image, 'caption': [str,...]}),
since this sandbox has no route to huggingface.co (same constraint as encoders.py's
real-checkpoint path). These verify __getitem__/collate/preprocessing logic; the
actual download and real Flickr30k caption diversity can only be confirmed once run
on Kaggle.
"""
import numpy as np
import torch
from PIL import Image

from real_captioning_data import FlickrCaptioningDataset, pil_to_tensor_01
from synthetic_data import collate_images_captions


class _FakeHFDataset:
    """Minimal stand-in matching HF `datasets.Dataset`'s interface (len + integer
    indexing returning a dict) with the confirmed nlphuji/flickr30k schema."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _make_fake_dataset(n=6):
    items = []
    for i in range(n):
        # vary size, aspect ratio, and mode across examples -- real Flickr30k images
        # are arbitrary size/aspect and mostly-but-not-all RGB
        if i % 3 == 0:
            img = Image.new("L", (400, 300), color=100)  # grayscale, tests .convert("RGB")
        elif i % 3 == 1:
            img = Image.new("RGBA", (500, 500), color=(10, 200, 30, 255))  # alpha channel
        else:
            img = Image.new("RGB", (333, 500), color=(i * 20 % 255, 50, 80))
        captions = [f"caption variant {v} for image {i}" for v in range(5)]
        items.append({"image": img, "caption": captions})
    return _FakeHFDataset(items)


def test_pil_to_tensor_handles_non_rgb_modes_and_produces_correct_shape_range():
    for mode, size in [("L", (400, 300)), ("RGBA", (500, 500)), ("RGB", (333, 500))]:
        img = Image.new(mode, size, color=(128,) if mode == "L" else (10, 20, 30, 255)[:len(mode)])
        t = pil_to_tensor_01(img, 224)
        assert t.shape == (3, 224, 224), f"wrong shape for mode {mode}: {t.shape}"
        assert t.min() >= 0.0 and t.max() <= 1.0, f"values out of [0,1] range for mode {mode}"
        assert t.dtype == torch.float32


def test_dataset_getitem_shape_and_range():
    ds = FlickrCaptioningDataset(hf_dataset=_make_fake_dataset(n=6), image_size=224, seed=0)
    assert len(ds) == 6
    image_tensor, caption = ds[0]
    assert image_tensor.shape == (3, 224, 224)
    assert image_tensor.min() >= 0.0 and image_tensor.max() <= 1.0
    assert isinstance(caption, str)
    assert "image 0" in caption  # sampled from image 0's own caption list, not another image's


def test_caption_sampling_exercises_genuine_ambiguity():
    """Confirms repeated access to the SAME image index can return DIFFERENT valid
    captions -- the actual property this dataset exists to exercise, unlike
    synthetic_data.py's exact one-to-one mapping."""
    ds = FlickrCaptioningDataset(hf_dataset=_make_fake_dataset(n=1), image_size=64, seed=0)
    seen_captions = set()
    for _ in range(50):
        _, caption = ds[0]
        seen_captions.add(caption)
    assert len(seen_captions) > 1, (
        f"Expected multiple distinct captions sampled across repeated accesses to the "
        f"same image (5 available), got only {seen_captions} -- sampling isn't "
        f"actually varying."
    )
    for c in seen_captions:
        assert "image 0" in c, f"Got a caption belonging to a different image: {c}"


def test_collate_images_captions_is_compatible():
    """Confirms the existing (already-tested, generic) collate function from
    synthetic_data.py works as a drop-in for this new dataset with no modification."""
    ds = FlickrCaptioningDataset(hf_dataset=_make_fake_dataset(n=4), image_size=224, seed=0)
    batch = [ds[i] for i in range(4)]
    images, captions = collate_images_captions(batch)
    assert images.shape == (4, 3, 224, 224)
    assert len(captions) == 4
    assert all(isinstance(c, str) for c in captions)


def test_dataloader_end_to_end():
    """Full DataLoader integration, matching exactly how train.py will actually use
    this class."""
    from torch.utils.data import DataLoader
    ds = FlickrCaptioningDataset(hf_dataset=_make_fake_dataset(n=8), image_size=224, seed=0)
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_images_captions, shuffle=True)
    batches = list(dl)
    assert len(batches) == 2
    for images, captions in batches:
        assert images.shape == (4, 3, 224, 224)
        assert len(captions) == 4
