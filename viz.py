'''Train PS-KD: learning with PyTorch.'''
from __future__ import print_function

#----------------------------------------------------
#  Pytorch
#----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from loader import custom_dataloader

#--------------
#  Datalodader
#--------------
from loader import custom_dataloader

#----------------------------------------------------
#  Load CNN-architecture
#----------------------------------------------------
from models.network import get_network

#--------------
# Util
#--------------
from utils.dir_maker import DirectroyMaker
from utils.AverageMeter import AverageMeter
from utils.metric import metric_ece_aurc_eaurc
from utils.color import Colorer
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults

#----------------------------------------------------
#  Etc
#----------------------------------------------------
import os, logging
import argparse
import numpy as np
import json

#----------------------------------------------------
#  Training Setting parser
#----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='visualizer for class prototype vectors')
    parser.add_argument('--experiment_dir', type=str, default='expts',
                        help='Directory name where the model ckpts are stored')
    args = parser.parse_args()
    return parser, args

#----------------------------------------------------
#  Colour print 
#----------------------------------------------------
C = Colorer.instance()

def main():
    parser, args = parse_args()
    config_file = os.path.join(args.experiment_dir, 'config/config.json')
    model_dir = os.path.join(args.experiment_dir, 'model')
    assert os.path.exists(config_file), "config file path incorrect"
    assert os.path.exists(model_dir), "model directory path incorrect"
    
    config_dict = json.load(open(config_file, 'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(config_dict)
    config_args = parser.parse_args(namespace=t_args)
    config_args.gpu = 0
    net = get_network(config_args)
    net = net.cuda()
    net.eval()
    start_epoch = config_args.start_epoch
    end_epoch = config_args.end_epoch
    saveckp_freq = config_args.saveckp_freq
    train_loader, valid_loader, train_sampler = custom_dataloader.dataloader(config_args)
    sim_matrices = []
    learnable_parameters = []
    teacher_before_learnable = []
    embeddings = []
    teacher_after_learnable = []
    
    for epoch in range(start_epoch, end_epoch):
        if (epoch+1) % saveckp_freq != 0:
            continue
        inputs, targets, input_indices = next(iter(train_loader))
        inputs = inputs.cuda(config_args.gpu, non_blocking=True)
        targets = targets.cuda(config_args.gpu, non_blocking=True)
        checkpoint = torch.load(os.path.join(model_dir, f'checkpoint_{epoch:03d}.pth'))
        net.load_state_dict(checkpoint['net'])
        learnable_params = net.learnable_params.weight.data  # tensor of shape = [num_classes, 512]
        learnable_params = learnable_params.clone().detach().cpu().numpy()  # nparray of shape = [num_classes, 512]
        learnable_parameters.append(learnable_params)
        similarity_matrix = learnable_params @ learnable_params.T
        sim_matrices.append(similarity_matrix)
        embedding = net(inputs)
        detached_embedding = embedding.clone().detach()
        embeddings.append(detached_embedding.cpu().numpy())
        teacher_output_before_learnable = F.normalize(net.teacher_head(detached_embedding))
        teacher_before_learnable.append(teacher_output_before_learnable.clone().detach().cpu().numpy())
        teacher_after_learnable.append(net.learnable_params(teacher_output_before_learnable).clone().detach().cpu().numpy())
    
    sim_matrices = np.array(sim_matrices)
    learnable_parameters = np.array(learnable_parameters)
    teacher_before_learnable = np.array(teacher_before_learnable)
    embeddings = np.array(embeddings)
    teacher_after_learnable = np.array(teacher_after_learnable)
    np.save(open(os.path.join(model_dir, 'teacher_logits.npy'), 'wb'), teacher_after_learnable)
    np.save(open(os.path.join(model_dir, 'embeddings.npy'), 'wb'), embeddings)
    np.save(open(os.path.join(model_dir, 'learnable_parameters_similarity.npy'), 'wb'), sim_matrices)
    np.save(open(os.path.join(model_dir, 'learnable_parameters.npy'), 'wb'), learnable_parameters)
    np.save(open(os.path.join(model_dir, 'teacher_output_before_learnable.npy'), 'wb'), teacher_before_learnable)

if __name__ == '__main__':
    main()
