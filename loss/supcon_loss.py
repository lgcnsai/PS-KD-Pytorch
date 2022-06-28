import torch.nn as nn
import torch


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    def forward(self, teacher_output, student_output):
        """Compute loss for model.
        Args:
            teacher_output: vector of shape [bsz, n_classes].
            student_output: vector of shape [bsz, n_classes].
        Returns:
            A loss scalar.
        """

        # todo normalize both the teacher and the student output
        # question is normalize in what way

        # todo put in temperature into the softmax?
        # todo kill gradients if teacher and student output are too similar

        loss_per_minibatch_sample = self.loss_fn(input=teacher_output, target=student_output.softmax(dim=1))
        z = torch.zeros((), device=loss_per_minibatch_sample.device, dtype=loss_per_minibatch_sample.dtype)
        loss_per_minibatch_sample_killed_gradients = torch.where(loss_per_minibatch_sample < 0.1,
                                                                 loss_per_minibatch_sample, z)
        loss = loss_per_minibatch_sample_killed_gradients.mean()
        return loss