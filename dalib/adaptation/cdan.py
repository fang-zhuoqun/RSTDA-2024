from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules.classifier import Classifier as ClassifierBase
from common.utils.metric import binary_accuracy
from ..modules.grl import WarmStartGradientReverseLayer
from ..modules.entropy import entropy


__all__ = ['ConditionalDomainAdversarialLoss', 'ImageClassifier']


class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The Conditional Domain Adversarial Loss used in `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_

    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(T(f_j^t, g_j^t))],\\

    where :math:`T` is a :class:`MultiLinearMap`  or :class:`RandomizedMultiLinearMap` which convert two tensors to a single tensor.

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        num_classes (int, optional): Number of classes. Default: -1
        features_dim (int, optional): Dimension of input features. Default: -1
        randomized_dim (int, optional): Dimension of features after randomized. Default: 1024
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.

    Inputs:
        - g_s (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, )`.

    Examples::

        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
        >>> import torch
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim * num_classes, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
    """

    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: Optional[bool] = False,
                 randomized: Optional[bool] = False, num_classes: Optional[int] = -1,
                 features_dim: Optional[int] = -1, randomized_dim: Optional[int] = 1024,
                 reduction: Optional[str] = 'mean',
                 max_iters_warmup: Optional[int] = 1000, eps = 0.0):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=max_iters_warmup, auto_step=True)
        self.entropy_conditioning = entropy_conditioning
        self.eps = eps
        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()

        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        # d_label = torch.cat((
        #     torch.ones((g_s.size(0), 1)).to(g_s.device),
        #     torch.zeros((g_t.size(0), 1)).to(g_t.device),
        # ))
        d_label = torch.cat((
            torch.ones((g_s.size(0), 1)).to(g_s.device) * (1-self.eps),
            torch.ones((g_t.size(0), 1)).to(g_t.device) * self.eps,
        ))
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        # self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
        self.domain_discriminator_accuracy = binary_accuracy(d, torch.where(d_label>0.5, 1, 0))
        return self.bce(d, d_label, weight.view_as(d))


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map

    Given two inputs :math:`f` and :math:`g`, the definition is

    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),

    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.

    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024

    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


# Add new Classifier from TAADA.
import math


# class BaseClassifier(nn.Module):
#     def __init__(self, out_features: int, num_classes: int, bottleneck_dim: Optional[int] = 152, training=False, dropout_p=0.5):
#         super().__init__()
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(out_features, bottleneck_dim),
#             nn.BatchNorm1d(bottleneck_dim, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout_p)
#         )
#
#         self.nn1 = nn.Linear(bottleneck_dim, num_classes)
#
#         self.training = training
#         self.dropout_p = dropout_p
#         # Add for weight initialization.
#         for m in self.modules():
#             classname = m.__class__.__name__
#             if classname.find('Conv') != -1:
#                 m.weight.data.normal_(0.0, 0.01)
#                 m.bias.data.normal_(0.0, 0.01)
#             elif classname.find('BatchNorm') != -1:
#                 m.weight.data.normal_(1.0, 0.01)
#                 m.bias.data.fill_(0)
#             elif classname.find('Linear') != -1:
#                 m.weight.data.normal_(0.0, 0.01)
#                 m.bias.data.normal_(0.0, 0.01)
#
#     def forward(self, x):
#         f = self.fc1(x)
#         if self.training:
#             f.mul_(math.sqrt(1 - self.dropout_p))
#         predictions = self.nn1(f)
#         return predictions, f


class ResClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 152,
                 training=False, dropout_p=0.5, **kwargs):
        super(ResClassifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )

        self.head = nn.Linear(bottleneck_dim, num_classes)

        self.training = training
        self.dropout_p = dropout_p
        self.backbone = backbone
        self.num_classes = num_classes
        self._features_dim = bottleneck_dim
        self.bottleneck = nn.Identity()
        # Add for weight initialization.
        # for m in self.fc1.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0.0, 0.01)
        #         m.bias.data.normal_(0.0, 0.01)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(1.0, 0.01)
        #         m.bias.data.fill_(0)
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0.0, 0.01)
        #         m.bias.data.normal_(0.0, 0.01)
        # nn.init.normal_(self.head.weight, 0.0, 0.01)
        # nn.init.normal_(self.head.bias, 0.0, 0.01)

    def forward(self, x):
        x = self.backbone(x)
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))  # L2 normalize Dropout.
        f = self.bottleneck(fc1_emb)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions


# class ResClassifier(nn.Module):
#     def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 152,
#                  training=False, dropout_p=0.5, **kwargs):
#         super(ResClassifier, self).__init__()
#
#         self.head = BaseClassifier(backbone.out_features, num_classes, bottleneck_dim, training, dropout_p)
#
#         self.training = training
#         self.backbone = backbone
#         self.bottleneck = nn.Identity()
#         self._features_dim = bottleneck_dim
#
#
#     def forward(self, x):
#         x = self.backbone(x)
#         predictions, f = self.head(x)
#         if self.training:
#             return predictions, f
#         else:
#             return predictions

    @property
    def features_dim(self):
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
