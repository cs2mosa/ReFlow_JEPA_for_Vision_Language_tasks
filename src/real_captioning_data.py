"""
Real image-captioning dataset: nlphuji/flickr30k (~31k images, 5 human captions each)
-- the point where genuine multimodal ambiguity (multiple valid captions per image)
becomes testable, unlike synthetic_data.py's exact one-to-one template mapping.

Schema confirmed via a real, working usage example found via search (not guessed):
    item['image']    -- native PIL Image, auto-decoded by `datasets` (embedded in the
                         dataset's parquet files -- no separate download/path
                         resolution step needed)
    item['caption']  -- list of 5 human-written caption strings for that image

Requires internet access to download on first use (this sandbox has none -- see
encoders.py's own docstring for the identical constraint). FlickrCaptioningDataset
accepts an optional pre-loaded `hf_dataset` argument specifically so its
__getitem__/collate logic can be unit-tested here (test_15) with a local stand-in
matching the confirmed schema, without needing the real download.
"""
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def pil_to_tensor_01(image: Image.Image, size: int) -> torch.Tensor:
    """Converts a PIL image (arbitrary mode/size) to a [0,1]-range float tensor,
    (3, size, size), matching synthetic_data.py's own raw-pixel convention -- real
    per-channel normalization (mean/std) is applied downstream in
    ReflowJEPA._visual_forward, not here. .convert("RGB") handles grayscale ('L') and
    RGBA images robustly -- Flickr30k, like most real photo datasets, contains a small
    number of non-RGB images, and skipping this would crash or silently corrupt those."""
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class FlickrCaptioningDataset(Dataset):
    def __init__(self, hf_split: str = "test", karpathy_split_filter: str = None,
                 image_size: int = 224, seed: int = 0, hf_dataset=None):
        """
        hf_split: nlphuji/flickr30k's own physical HF `datasets` split name. Confirmed
            via a real usage example that the ENTIRE ~31k-image dataset lives under a
            single split literally named "test" (an unusual but real convention for
            this specific repo, likely because it was uploaded primarily as a
            retrieval-benchmark eval set) -- NOT the conventional train/val/test
            meaning.
        karpathy_split_filter: optional filter on the dataset's OWN internal 'split'
            COLUMN (train/val/test per the standard Karpathy Flickr30k partition),
            which is separate from the physical HF split above. None (default) uses
            every row -- reasonable for this exploratory stage, matching how
            SyntheticCaptioningDataset also doesn't enforce a train/eval split.
        hf_dataset: inject a pre-loaded dataset-like object (supporting __len__ and
            integer __getitem__ returning {"image": PIL.Image, "caption": [str,...]})
            to bypass the real download -- used by this project's own tests to verify
            __getitem__/collate logic without internet access.
        """
        if hf_dataset is not None:
            self.ds = hf_dataset
        else:
            from datasets import load_dataset
            self.ds = load_dataset("nlphuji/flickr30k", split=hf_split)
            if karpathy_split_filter is not None:
                self.ds = self.ds.filter(lambda ex: ex["split"] == karpathy_split_filter)
        self.image_size = image_size
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image_tensor = pil_to_tensor_01(item["image"], self.image_size)
        # Genuine multimodal ambiguity, unlike synthetic_data.py's exact one-to-one
        # mapping: randomly sample ONE of the 5 human captions per access. A
        # DIFFERENT valid caption may be returned for the SAME image across different
        # epochs/steps -- directly exercises the stochastic source's theoretical
        # motivation (Z_0's noise can, in principle, resolve to different valid
        # targets for the same conditioning image) in a way the synthetic dataset's
        # exact one-to-one mapping never could.
        caption = self.rng.choice(item["caption"])
        return image_tensor, caption
