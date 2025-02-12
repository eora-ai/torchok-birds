from typing import Any, Dict, List

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torchok.constructor import BACKBONES, DETECTION_NECKS, HEADS, TASKS
from torchok.constructor.config_structure import Phase
from torchok.models.backbones import BackboneWrapper
from torchok.tasks.base import BaseTask


@TASKS.register_class
class SingleStageDetectionTask(BaseTask):
    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            head_name: str,
            neck_name: str = None,
            num_scales: int = None,
            backbone_params: dict = None,
            neck_params: dict = None,
            head_params: dict = None,
            **kwargs
    ):
        """Init SingleStageDetectionTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            backbone_name: name of the backbone architecture in the BACKBONES registry.
            neck_name: name of the head architecture in the DETECTION_NECKS registry.
            head_name: name of the neck architecture in the HEADS registry.
            num_scales: number of feature maps that will be passed from backbone to the neck
                starting from the last one.
                Example: for backbone output `[layer1, layer2, layer3, layer4]` and `num_scales=3`
                neck will get `[layer2, layer3, layer4]`.
            backbone_params: parameters for backbone constructor.
            neck_params: parameters for neck constructor. `in_channels` will be set automatically based on backbone.
            head_params: parameters for head constructor. `in_channels` will be set automatically based on neck.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, **kwargs)

        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)
        self.num_scales = num_scales or len(self.backbone.out_encoder_channels)

        # NECK
        if neck_name is not None:
            neck_params = neck_params or dict()
            neck_in_channels = self.backbone.out_encoder_channels[-self.num_scales:][::-1]
            self.neck = DETECTION_NECKS.get(neck_name)(in_channels=neck_in_channels, **neck_params)
            head_in_channels = self.neck.out_channels
        else:
            self.neck = nn.Identity()
            head_in_channels = self.backbone.out_encoder_channels[-self.num_scales:][::-1]

        # HEAD
        head_params = head_params or dict()
        # MMDet detection heads may change losses, but they don't attach JointLoss instance internally.
        # Therefore, it is required to provide JointLoss instance to bbox_head.loss function.
        self.bbox_head = HEADS.get(head_name)(joint_loss=self.losses, in_channels=head_in_channels, **head_params)

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Forward method.

        Args:
            x: tensor of shape `(B, C, H, W)`. Batch of input images.

        Returns:
            List of length B containing dicts with two items `bboxes` and `labels`.

            - `bboxes` (torch.Tensor):
                tensor of shape `(N, 5)`, where N is the number of bboxes on the image, may be different for
                each image and even may be 0. Each box is form `[x1, y1, x2, y2, confidence]`.
            - `labels` (torch.Tensor):
                tensor of shape `(N)`, containing class label of each bbox.
        """
        features = self.backbone.forward_features(x)[-self.num_scales:]
        features = self.neck(features)
        features = self.bbox_head(features)
        output = self.bbox_head.format_dict(features)
        output = self.bbox_head.get_bboxes(**output, img_metas=[dict(img_shape=x.shape[-2:])] * x.shape[0])
        return output

    def forward_with_gt(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Forward with ground truth labels.

        Args:
            batch: Dictionary with the following keys and values:

                - `image` (torch.Tensor):
                    tensor of shape `(B, C, H, W)`, representing input images.
                - `bboxes` (List[torch.Tensor]):
                    list of B tensors of shape `(N, 4)`, where N is the number of bboxes on the image, may be different
                    for each image and even may be 0. Each box is form `[x_left, y_top, x_right, y_bottom]`. May absent.
                - `labels` (List[torch.Tensor]):
                    list of B tensors of shape `(N)`, containing class label of each bbox. May absent.

        Returns:
            Dictionary with the keys related to specific detection head, input image shape
            and ground truth values if present.
        """
        input_data = batch.get('image')
        img_shape = (*input_data.shape[-2:], input_data.shape[-3])
        img_metas = [dict(orig_img_shape=orig_shape, img_shape=img_shape) for orig_shape in batch.get("orig_img_shape")]

        features = self.backbone.forward_features(input_data)[-self.num_scales:]
        neck_out = self.neck(features)
        if self.bbox_head.requires_meta_in_forward:
            prediction = self.bbox_head(neck_out, img_metas)
        else:
            prediction = self.bbox_head(neck_out)
        output = self.bbox_head.format_dict(prediction)

        output['img_metas'] = img_metas

        if 'bboxes' in batch:
            output['gt_bboxes'] = batch.get('bboxes')
            output['gt_labels'] = batch.get('label')

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(BackboneWrapper(self.backbone), self.neck, self.bbox_head)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete training loop."""
        output = self.forward_with_gt(batch)
        total_loss, tagged_loss_values = self.bbox_head.loss(self.losses, **output)

        output['prediction'] = self.bbox_head.get_bboxes(**output)
        output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        self.metrics_manager.update(Phase.TRAIN, **output)
        output_dict = {'loss': total_loss}
        output_dict.update(tagged_loss_values)
        return output_dict

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int,
                        dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)

        output['prediction'] = self.bbox_head.get_bboxes(**output)
        output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        self.metrics_manager.update(Phase.VALID, dataloader_idx, **output)

        if self._hparams.task.compute_loss_on_valid:
            total_loss, tagged_loss_values = self.bbox_head.loss(self.losses, **output)
            output_dict = {'loss': total_loss}
            output_dict.update(tagged_loss_values)
        else:
            output_dict = {}

        return output_dict

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch)
        output['prediction'] = self.bbox_head.get_bboxes(**output)
        output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        self.metrics_manager.update(Phase.TEST, dataloader_idx, **output)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete predict loop."""
        output = self.forward_with_gt(batch)
        output['prediction'] = self.bbox_head.get_bboxes(**output)
        if 'gt_bboxes' in output:
            output['target'] = [dict(bboxes=bb, labels=la) for bb, la in zip(output['gt_bboxes'], output['gt_labels'])]
        return output
