import torch.nn as nn
import torch
import torch.nn.functional as F


class StudentLoss(nn.Module):
    def __init__(self, temperature=1, base_temperature=1.0):
        super(StudentLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, student_logits, ground_truth, teacher_logits, lin_comb_alpha):
        """Compute loss for model.
        Args:
            teacher_logits: vector of shape [batch, n_classes], 
            ground_truth: one hot labels, array of shape [batch, n_classes].
            student_logits: vector of shape [batch, n_classes].
        Returns:
            A loss scalar.
        """
        # Student loss: crossentropy (student output (logits -> temperature softmax)
        # compared to dot products of ground truth and teacher prediction
        # (logits -> temperature softmax temperature > 1) (output of last linear teacher layer)
        
        soft_label = ((1 - lin_comb_alpha) * ground_truth) + (lin_comb_alpha * F.softmax(teacher_logits *
                                                                                       self.temperature, dim=1)).cuda()
        loss = self.loss_fn(input=student_logits * self.temperature, target=soft_label)
        return loss


class TeacherLoss(nn.Module):
    def __init__(self, temperature=1.0, kill_gradients=False):
        super(TeacherLoss, self).__init__()
        self.temperature = temperature
        self.kill_gradients = kill_gradients
        self.sim_threshold = 0.9
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    def forward(self, teacher_logits, ground_truth):
        # Teacher_logits: [batch, n_classes]
        # ground_truth: [batch,] , not one hot labels
        # todo kill gradients if teacher output and learnable params are too similar
        loss = self.loss_fn(input = teacher_logits * self.temperature, target=ground_truth)
        
        if self.kill_gradients:
            batch_size, n_classes = teacher_logits.shape
            indices = torch.arange(batch_size).cuda()
            gt_logits = teacher_logits[indices, ground_truth]
            sim_mask = (gt_logits > self.sim_threshold).detach()
            mask = torch.ones(batch_size, ).cuda()
            mask[sim_mask] = 0
            loss = mask * loss
        
        loss = torch.mean(loss)

        return loss

