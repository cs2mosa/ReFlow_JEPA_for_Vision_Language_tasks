"""
Real image-captioning dataset: Mozilla/flickr30k-transformed-captions (~31k real
photos, 5 original human captions each, plus an LLM-rewritten "de-biased" alt_text
column we deliberately do NOT use) -- the point where genuine multimodal ambiguity
(multiple valid captions per image) becomes testable, unlike synthetic_data.py's exact
one-to-one template mapping.

NOTE ON DATASET CHOICE: originally implemented against nlphuji/flickr30k, the more
commonly-cited Flickr30k HF repo. That failed in practice with "Dataset scripts are no
longer supported, but found flickr30k.py" -- nlphuji/flickr30k uses a legacy
custom-Python-loading-script format, and HF's `datasets` library has since (correctly,
for security reasons) dropped support for executing arbitrary loading scripts. This
repo is a "de-biased" derivative published in genuine Parquet format (confirmed via
its own listing: "Auto-converted to Parquet"), which loads with the modern library.
It preserves the same underlying 31,014 Flickr30k images and, in `original_alt_text`,
the same original 5 human captions per image -- we use that field, not the derived
`alt_text` column (LLM-rewritten for a different purpose: reducing certain annotator
biases -- not something we want introduced into a test of the base theory).

Schema confirmed via the dataset's own published dataset_info AND a real working code
snippet from a derivative dataset that indexes this exact field name:
    item['image']              -- native PIL Image (embedded, auto-decoded by `datasets`)
    item['original_alt_text']  -- list of 5 original human-written caption strings

Given this is the SECOND time a dataset's real-world field names/loading mechanism
differed from what looked authoritative in documentation, caption-field lookup below
is defensive: it tries several candidate names and fails with a clear, actionable
error (listing the actual available fields) rather than a bare KeyError deep in
training, if the schema has changed again by the time this runs.

Requires internet access to download on first use (this sandbox has none -- see
encoders.py's own docstring for the identical constraint). FlickrCaptioningDataset
accepts an optional pre-loaded `hf_dataset` argument specifically so its
__getitem__/collate logic can be unit-tested here (test_15) with a local stand-in,
without needing the real download.
"""
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_CAPTION_FIELD_CANDIDATES = ["original_alt_text", "caption", "captions", "alt_text", "sentences"]


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
        hf_split: this repo's own physical HF `datasets` split name. Confirmed via its
            published dataset_info that the ENTIRE ~31k-image dataset lives under a
            single split literally named "test" (an unusual but real convention,
            inherited from the base nlphuji/flickr30k repo it derives from, likely
            because it was uploaded primarily as a retrieval-benchmark eval set) --
            NOT the conventional train/val/test meaning.
        karpathy_split_filter: optional filter on the dataset's OWN internal 'split'
            COLUMN (train/val/test per the standard Karpathy Flickr30k partition),
            which is separate from the physical HF split above. None (default) uses
            every row -- reasonable for this exploratory stage, matching how
            SyntheticCaptioningDataset also doesn't enforce a train/eval split.
        hf_dataset: inject a pre-loaded dataset-like object (supporting __len__ and
            integer __getitem__ returning a dict with 'image' and one of
            _CAPTION_FIELD_CANDIDATES) to bypass the real download -- used by this
            project's own tests to verify __getitem__/collate logic without internet
            access.
        """
        if hf_dataset is not None:
            self.ds = hf_dataset
        else:
            from datasets import load_dataset
            self.ds = load_dataset("Mozilla/flickr30k-transformed-captions", split=hf_split)
            if karpathy_split_filter is not None:
                self.ds = self.ds.filter(lambda ex: ex["split"] == karpathy_split_filter)
        self.image_size = image_size
        self.rng = random.Random(seed)

        # Defensive field-name detection: this is the SECOND time this dataset's
        # real-world schema differed from what looked authoritative beforehand
        # (nlphuji/flickr30k's script-based loading being dropped by the library was
        # the first). Fail loudly and specifically here rather than with a bare
        # KeyError deep inside training's first __getitem__ call.
        first_item = self.ds[0]
        self._caption_field = next(
            (f for f in _CAPTION_FIELD_CANDIDATES if f in first_item), None)
        if self._caption_field is None:
            raise KeyError(
                f"Could not find a recognized caption field in this dataset. Tried: "
                f"{_CAPTION_FIELD_CANDIDATES}. Actual available fields: "
                f"{list(first_item.keys())}. Add the correct field name to "
                f"_CAPTION_FIELD_CANDIDATES in real_captioning_data.py."
            )
        if "image" not in first_item:
            raise KeyError(
                f"Could not find an 'image' field in this dataset. Actual available "
                f"fields: {list(first_item.keys())}."
            )

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
        caption = self.rng.choice(item[self._caption_field])
        return image_tensor, caption

