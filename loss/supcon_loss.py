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
    def __init__(self, temperature=1.0, sim_threshold=0.9, dis_sim_threshold=0.2, kill_gradients=False):
        super(TeacherLoss, self).__init__()
        self.temperature = temperature
        self.kill_gradients = kill_gradients
        self.sim_threshold = sim_threshold  # 0.8 to 0.9, we can also grid-search here
        self.dis_sim_threshold = dis_sim_threshold  # don't set negative threshold to -0.9, set it larger
        # "the cosine should have a positive value like 0.3 or something like that" somewhere between 0.3 and 0.1
        #  -> grid-search
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    def forward(self, teacher_logits, ground_truth):
        # Teacher_logits: [batch, n_classes]
        # ground_truth: [batch,] , not one hot labels
        if self.kill_gradients:
            identity_matrix = torch.eye(teacher_logits.shape[1], dtype=torch.bool, device=teacher_logits.device)  # [n_classes, n_classes]
            ground_truth_mask = identity_matrix[ground_truth]  # [batch, n_classes]
            wrong_pred_mask = torch.ones_like(teacher_logits, dtype=torch.bool) ^ ground_truth_mask  # XOR
            dis_sim_setter = torch.ones((), device=teacher_logits.device, dtype=teacher_logits.dtype) * -1
            sim_setter = torch.ones((), device=teacher_logits.device, dtype=teacher_logits.dtype) * self.sim_threshold
            # set large similarities to the threshold if they are correct predictions
            teacher_logits[ground_truth_mask] = torch.where(teacher_logits[ground_truth_mask] > self.sim_threshold,
                                                            sim_setter, teacher_logits[ground_truth_mask])
            # set small similarities to -1 if they are incorrect predictions
            teacher_logits[wrong_pred_mask] = torch.where(teacher_logits[wrong_pred_mask] < self.dis_sim_threshold,
                                                          dis_sim_setter, teacher_logits[wrong_pred_mask])

        loss = self.loss_fn(input=teacher_logits * self.temperature, target=ground_truth)
        
        #if self.kill_gradients:
        #    batch_size, n_classes = teacher_logits.shape
        #    indices = torch.arange(batch_size).cuda()
        #    gt_logits = teacher_logits[indices, ground_truth]
        #    sim_mask = (gt_logits > self.sim_threshold).detach()
        #    mask = torch.ones(batch_size, ).cuda()
        #    mask[sim_mask] = 0
        #    loss = mask * loss
        
        loss = torch.mean(loss)

        return loss

