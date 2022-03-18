
# Apache 2.0 License
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.


""" This file was copied from: https://github.com/facebookresearch/deit/blob/main/losses.py
    
    and modified by Verena Battis, Fraunhofer SIT in order to use the DeiT as attack model in a model extraction attack.

    Modified passages are marked as follows: 

    #### Begin modifications 

    Code added or modified

    #### End modifications  
    
    
    Apache 2.0 License
    Copyright (c) 2022, Fraunhofer e.V.
    All rights reserved.
    
"""


"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a target model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, target_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, softmax_target: bool, output_dir: str):
        super().__init__()
        self.base_criterion = base_criterion
        self.target_model = target_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.softmax_target = softmax_target
        self.output_dir = output_dir

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the target model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs

        if self.distillation_type == 'none':
            return self.base_criterion(outputs, labels)
        base_loss = self.base_criterion(outputs, labels) if self.alpha != 1 else 0

        if outputs_kd is None:
            outputs_kd = outputs
            # raise ValueError("When knowledge distillation is enabled, the model is "
            #                  "expected to return a Tuple[Tensor, Tensor] with the output of the "
            #                  "class_token and the dist_token")
        # don't backprop through the target
        with torch.no_grad():
            target_outputs = self.target_model(inputs)
            if self.output_dir:
                target_predictions = target_outputs.cpu().argmax(1).numpy()
                with (self.output_dir / "target_predictions.log").open("a") as f:
                    f.write(' '.join(map(str, target_predictions)) + '\n')

        if self.distillation_type == 'soft':
            T = self.tau
            if self.softmax_target:
                target_outputs = torch.log(target_outputs + 1e-10)  # add small constant to avoid log 0
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(target_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, target_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

#### Begin modifications 
    
class KLDivLossWithTemperature(torch.nn.Module):
    def __init__(self, tau=1, softmax_target=False):
        super().__init__()
        self.tau = tau
        self.softmax_target = softmax_target

    def forward(self, outputs, targets):
        T = self.tau
        if self.softmax_target:
            targets = torch.log(targets + 1e-10)  # add small constant to avoid log 0
        distillation_loss = F.kl_div(
            F.log_softmax(outputs / T, dim=1),
            F.log_softmax(targets / T, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T) / outputs.numel()
        return distillation_loss

#### End modifications 
