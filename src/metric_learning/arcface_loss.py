import torch.nn as nn_torch
from torch.nn import functional
import torch


class AngularPenaltySMLoss(nn_torch.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, scale=None, margin=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.scale = 64.0 if not scale else scale
            self.margin = 0.5 if not margin else margin
        if loss_type == 'sphereface':
            self.scale = 64.0 if not scale else scale
            self.margin = 1.35 if not margin else margin
        if loss_type == 'cosface':
            self.scale = 30.0 if not scale else scale
            self.margin = 0.4 if not margin else margin
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn_torch.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, inputs, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(inputs) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for parameters in self.fc.parameters():
            parameters = functional.normalize(parameters, p=2, dim=1)

        inputs = functional.normalize(inputs, p=2, dim=1)

        original_logits = self.fc(inputs)  # Выход слоя классификации
        if self.loss_type == 'cosface':
            numerator = self.scale * \
                (torch.diagonal(original_logits.transpose(0, 1)[labels]) - self.margin)
        if self.loss_type == 'arcface':
            numerator = self.scale * \
                torch.cos(torch.acos(torch.clamp(torch.diagonal(
                    original_logits.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.margin)
        if self.loss_type == 'sphereface':
            numerator = self.scale * \
                torch.cos(
                    self.margin * torch.acos(torch.clamp(torch.diagonal(original_logits.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        marginal_logits = torch.cat([torch.cat((original_logits[i, :y], original_logits[i, y+1:])).unsqueeze(0)
                            for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * marginal_logits), dim=1)
        loss = numerator - torch.log(denominator)
        return -torch.mean(loss)
