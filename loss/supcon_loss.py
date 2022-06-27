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
        print(teacher_output.shape)
        print(student_output.shape)
        print(teacher_output.dtype)
        print(student_output.dtype)
        loss = self.loss_fn(input=teacher_output, target=student_output.softmax(dim=1))
        return loss