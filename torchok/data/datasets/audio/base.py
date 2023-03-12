import re
from pathlib import Path
from typing import Any, Optional, Union, Tuple

import numpy as np
import soundfile as sf
import pandas as pd
import torch
from torch.utils.data import Dataset
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.constructor import DATASETS


@DATASETS.register_class
class AudioClassificationDataset(Dataset):
    def __init__(self,
                 data_folder: str,
                 csv_path: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 period: int = 20,
                 num_classes: int = None,
                 input_column: str = 'audio_path',
                 input_dtype: str = 'float32',
                 target_column: str = 'label',
                 target_dtype: str = 'long',
                 valid_mode: bool = False,
                 test_mode: bool = False,):

        self.data_folder = Path(data_folder)
        self.num_classes = num_classes
        self.input_column = input_column
        self.target_column = target_column
        self.target_dtype = target_dtype
        self.augment = augment
        self.transform = transform
        self.period = period
        self.valid_mode = valid_mode
        self.test_mode = test_mode
        self.input_dtype = input_dtype

        csv_path = self.data_folder / csv_path
        self.csv = pd.read_csv(csv_path)
        

    def get_raw(self, idx: int) -> dict:
        """Get item sample without transform application.

        Returns:
            sample: dict, where
            sample['audio'] - np.array, representing image after augmentations.
            sample['target'] - Target class or labels.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        
        record = self.csv.iloc[idx]
        
        sample = {'index': idx}
        if not self.test_mode:
            target = np.zeros(self.num_classes, dtype=float)
            target[record[self.target_column]] = 1.0
            sample['target'] = target

        wav_name = record[self.input_column]
        y, sr = sf.read(self.data_folder / wav_name)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            if not self.valid_mode:
                start = np.random.randint(effective_length - len_y)
            else:
                start = 0
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            if not self.valid_mode:
                start = np.random.randint(len_y - effective_length)
            else:
                start = 0
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        y = np.nan_to_num(y)

        sample["data"] = y
        sample = self._apply_transform(self.augment, sample)
        sample["data"] = np.nan_to_num(sample["data"])

        return sample

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['target'] - Target class or labels, dtype=target_dtype.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)
        sample["data"] = torch.tensor(sample["data"], dtype=torch.__dict__[self.input_dtype])

        if not self.test_mode:
            sample['target'] = torch.tensor(sample['target']).type(torch.__dict__[self.target_dtype])

        return sample

    def _apply_transform(self, transform: Union[BasicTransform, BaseCompose], sample: dict) -> dict:
        """Is transformations based on API of albumentations library.

        Args:
            transform: Transformations from `albumentations` library.
                https://github.com/albumentations-team/albumentations/
            sample: Sample which the transformation will be applied to.

        Returns:
            Transformed sample.
        """
        if transform is None:
            return sample

        new_sample = transform(**sample)
        return new_sample

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.csv)
