from typing import Dict, Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation

from torchok.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
from torchok.tasks.base import BaseTask


class Preprocess(nn.Module):
    def __init__(self, spectrogram_params: Dict[str, Any], logmel_params: Dict[str, Any],
                 spec_aug_params: Dict[str, Any], use_spec_aug: bool = True, use_batch_norm: bool = True):
        super().__init__()
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(**spectrogram_params)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(**logmel_params)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(**spec_aug_params)
        # self.spectrogram_extractor = Spectrogram(n_fft=CFG.n_fft, hop_length=CFG.hop_length,
        #                                     win_length=CFG.n_fft, window="hann", center=True, pad_mode="reflect",
        #                                     freeze_parameters=True)
        # self.logmel_extractor = LogmelFilterBank(sr=CFG.sample_rate, n_fft=CFG.n_fft,
        #                                             n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax, ref=1.0, amin=1e-10, top_db=None,
        #                                             freeze_parameters=True)
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        #                                        freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(logmel_params.n_mels)
        self.use_batch_norm = use_batch_norm
        self.use_spec_aug = use_spec_aug

    def forward(self, x):
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        if self.use_batch_norm:
            x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.use_spec_aug:
            x = self.spec_augmenter(x)

        x = x.transpose(2, 3)

        return x, frames_num


@TASKS.register_class
class AudioClassificationTask(BaseTask):
    """A class for image classification task."""

    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            head_name: str = None,
            backbone_params: dict = None,
            head_params: dict = None,
            preprocess_params: dict = None,
            inputs: dict = None
    ):
        """Init ClassificationTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            pooling_name: name of the backbone architecture in the POOLINGS registry.
            head_name: name of the neck architecture in the HEADS registry.
            neck_name: if present, name of the head architecture in the NECKS registry. Otherwise, model will be created
                without neck.
            backbone_params: parameters for backbone constructor.
            neck_params: parameters for neck constructor. `in_channels` will be set automatically based on backbone.
            pooling_params: parameters for neck constructor. `in_channels` will be set automatically based on neck or
                backbone if neck is absent.
            head_params: parameters for head constructor. `in_channels` will be set automatically based on neck.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, inputs)
        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        self.preprocess = Preprocess(**preprocess_params)

        # HEAD
        head_params = head_params or dict()
        self.head = HEADS.get(head_name)(in_channels=self.backbone.out_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            x: torch.Tensor of shape `(B, C, H, W)`. Batch of input images.

        Returns:
            torch.Tensor of shape `(B, num_classes)`, representing logits per each image.
        """
        x, frames_num = self.preprocess(x)
        x = self.backbone(x)
        x = self.head(x, frames_num)["logit"]
        return x

    def forward_with_gt(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Tensor]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image` (torch.Tensor):
                    tensor of shape `(B, C, H, W)`, representing input images.
                - `target` (torch.Tensor):
                    tensor of shape `(B)`, target class or labels per each image.

        Returns:
            Dictionary with the following keys and values

            - 'embeddings': torch.Tensor of shape `(B, num_features)`, representing embeddings per each image.
            - 'prediction': torch.Tensor of shape `(B, num_classes)`, representing logits per each image.
            - 'target': torch.Tensor of shape `(B)`, target class or labels per each image. May absent.
        """
        input_data = batch.get('data')
        target = batch.get('target')
        x, frames_num = self.preprocess(input_data)
        x = self.backbone(x)
        output = self.head(x, frames_num)

        if target is not None:
            output['target'] = target

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(self.preprocess, self.backbone, self.head)
