'''Train PS-KD: learning with PyTorch.'''
from __future__ import print_function

#----------------------------------------------------
#  Pytorch
#----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
    parser.add_argument('--experiment_dir', type=str, default='expts',help='Directory name where the model ckpts are stored')
    args = parser.parse_args()
    return parser,args

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
    
    config_dict = json.load(open(config_file,'r'))
    t_args = argparse.Namespace()
    t_args.__dict__.update(config_dict)
    config_args = parser.parse_args(namespace=t_args)
    config_args.gpu = 0
    net = get_network(config_args)
    net = net.cuda()
    start_epoch = config_args.start_epoch
    end_epoch = config_args.end_epoch
    saveckp_freq = config_args.saveckp_freq
    
    sim_matrices = []
    
    for epoch in range(start_epoch, end_epoch):
        if (epoch+1) % saveckp_freq !=0 :
            continue
        checkpoint = torch.load(os.path.join(model_dir,'checkpoint_'+str(epoch)+'.pth'))  
        net.load_state_dict(checkpoint['net'])
        learnable_params = net.learnable_params.weight.data # tensor of shape = [num_classes, 512]
        learnable_params = learnable_params.clone().detach().cpu().numpy() # nparray of shape = [num_classes, 512]
        similarity_matrix = learnable_params @ learnable_params.T
        sim_matrices.append(similarity_matrix)
    
    sim_matrices = np.array(sim_matrices)
    np.save(open(os.path.join(model_dir,'learnable_parameters_similarity.npy'),'wb'),sim_matrices)

if __name__ == '__main__':
    main()