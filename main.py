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
#  Loss
#--------------
from loss.pskd_loss import Custom_CrossEntropy_PSKD
from loss.supcon_loss import StudentLoss, TeacherLoss

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


#----------------------------------------------------
#  Training Setting parser
#----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Online self-KD with soft labels')
    parser.add_argument('--lr', default=0.2, type=float, help='initial learning rate for student head and backbone')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--lr_decay_schedule', default=[150, 225], nargs='*', type=int, help='when to drop lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay for student head and backbone')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--end_epoch', default=300, type=int, help='number of training epoch to run')
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 128), this is the total'
                                                                    'batch size of all GPUs on the current node when '
                                                                    'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--experiments_dir', type=str, default='models',help='Directory name to save the model, log, config')
    parser.add_argument('--classifier_type', type=str, default='ResNet18', help='Select classifier')
    parser.add_argument('--data_path', type=str, default=None, help='download dataset path')
    parser.add_argument('--data_type', type=str, default=None, help='type of dataset')
    parser.add_argument('--alpha_T',default=0.8 ,type=float, help='alpha_T')
    parser.add_argument('--cosine_schedule', action='store_true', help='use cosine annealing learning rate schedule')
    parser.add_argument('--saveckp_freq', default=300, type=int, help='Save checkpoint every x epochs. Last model saving set to 299')
    parser.add_argument('--workers', default=40, type=int, help='number of workers for dataloader')
    parser.add_argument('--custom_transform', action='store_true', help='use supervised contrastive augmentation')
    parser.add_argument('--use_teacher_loss', action='store_true', help='backpropagate through teacher head')
    parser.add_argument('--use_student_loss', action='store_true', help='backpropagate through student head')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature')
    
    #parser.add_argument('--supervised_contrastive', action='store_true', help='add supervised contrastive loss to teacher output')
    parser.add_argument('--kill_similar_gradients', action='store_true',
                        help='kill gradients in teacher loss if the predictions are too similar and/or too dissimilar')
    parser.add_argument('--resume', type=str, default=None, help='load model path')
    parser.add_argument('--use_prior', action='store_true', help='use prior knowledge of superclasses')
    parser.add_argument('--sim_threshold', default=1.0, type=float, help='similarity threshold for teacher loss')
    parser.add_argument('--dis_sim_threshold', default=1.0, type=float, help='dissimilarity threshold for teacher loss')
    parser.add_argument('--teacher_lr', default=0.2, type=float,
                        help='learning rate for teacher head and learnable parameters')
    parser.add_argument('--teacher_weight_decay', default=1e-6, type=float,
                        help='weight decay for teacher head and learnable parameters')
    
    args = parser.parse_args()
    return check_args(args)

def check_args(args):
    # --epoch
    try:
        assert args.end_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args
    
#----------------------------------------------------
#  Adjust_learning_rate & get_learning_rate  
#----------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    warmup_length = 9
    if args.cosine_schedule:
        if epoch == 0:
            mult_factor = 0.01
        elif epoch <= warmup_length:
            # we will exponentially increase from 0.002 to 0.2 in the warmup-period
            mult_factor = np.power(100, 1/warmup_length)
        else:  # epoch > warmup_length
            # calculate the factor from previous epoch
            factor_previous = 0.5 * (1.0 + np.cos(np.pi * ((epoch - warmup_length - 1)/(args.end_epoch - warmup_length))))
            factor_now = 0.5 * (1.0 + np.cos(np.pi * ((epoch - warmup_length)/(args.end_epoch - warmup_length))))
            mult_factor = factor_now/factor_previous
    else:
        mult_factor = 1.
        for milestone in args.lr_decay_schedule:
            if epoch == milestone:
                mult_factor *= args.lr_decay_rate
            

    for param_group in optimizer.param_groups:
        param_group['lr'] *= mult_factor
    """
    lr = args.lr

    if args.cosine_schedule:
        t_cur = epoch
        t_end = args.end_epoch
        lr = 0.5 * lr * (1.0 + np.cos(np.pi * (t_cur / t_end)))
    else:
        for milestone in args.lr_decay_schedule:
            lr *= args.lr_decay_rate if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    """

        
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


#----------------------------------------------------
#  Top-1 / Top -5 accuracy
#----------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    
      


#----------------------------------------------------
#  Colour print 
#----------------------------------------------------
C = Colorer.instance()


def main(args=None):
    if args is None:
        args = parse_args()
    #----------------------------------------------------
    #  Prompt color print
    #----------------------------------------------------
    print(C.green("[!] Start the Online Self-KD with soft labels."))
    print(C.green("[!] Code borrowed from PSKD paper"))
    
    #-------------------------------------------------------------
    #  Create dir for saving experiments model, log, configuration
    #-------------------------------------------------------------
    dir_maker = DirectroyMaker(root=args.experiments_dir, save_model=True, save_log=True, save_config=True)
    model_log_config_dir = dir_maker.experiments_dir_maker(args)
    
    model_dir = model_log_config_dir[0]
    log_dir = model_log_config_dir[1]
    config_dir = model_log_config_dir[2]
    
    #----------------------------------------------------
    #  Save Configuration to config_dir
    #----------------------------------------------------
    paser_config_save(args, config_dir)    
    highest_acc = main_worker(0, None, model_dir, log_dir, args)
    print(C.green("[!] All Single GPU Training Done"))
    print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
    print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
    print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir))
    return highest_acc
        

def main_worker(gpu, ngpus_per_node, model_dir, log_dir, args):
    
    best_acc = 0
    net = get_network(args)
    args.gpu = gpu    
    #torch.cuda.set_device(args.gpu)
    net = net.cuda(args.gpu)
        
    set_logging_defaults(log_dir, args)

    #---------------------------------------------------
    #  Load Dataset
    #---------------------------------------------------
    train_loader, valid_loader, train_sampler = custom_dataloader.dataloader(args)
    
    #---------------------------------------------------
    #  Define loss function (criterion) and optimizer
    #----------------------------------------------------
    
    #if args.supervised_contrastive:
    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_student = StudentLoss(temperature=args.temperature).cuda(args.gpu)
    criterion_teacher = TeacherLoss(temperature=args.temperature, sim_threshold=args.sim_threshold,
                                    dis_sim_threshold=args.dis_sim_threshold,
                                    kill_gradients=args.kill_similar_gradients).cuda(args.gpu)
    #else:
    #    criterion_student = None
    #    criterion_teacher = None
    # use vicreg hyperparameters for the teacher head
    optimizer = torch.optim.SGD([
        {'params': net.conv1.parameters()},
        {'params': net.bn1.parameters()},  # not sure if this needs to be included to let gradients flow through
        {'params': net.layer1.parameters()},
        {'params': net.layer2.parameters()},
        {'params': net.layer3.parameters()},
        {'params': net.layer4.parameters()},
        {'params': net.student_head.parameters()},
        {'params': net.teacher_head.parameters(), "lr": args.teacher_lr, 'weight_decay': args.teacher_weight_decay},
        {'params': net.learnable_params.parameters(), "lr": args.teacher_lr, 'weight_decay': 0.0}
        # exclude learnable parameters from weight decay
    ],
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    #optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    #                            nesterov=True)

    #----------------------------------------------------
    #  load status & Resume Learning
    #----------------------------------------------------
    if args.resume:

        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        
        args.start_epoch = checkpoint['epoch'] + 1 
        alpha_t = checkpoint['alpha_t']
        best_acc = checkpoint['best_acc']
        #all_predictions = checkpoint['prev_predictions'].cpu()
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #print(C.green("[!] [Rank {}] Model loaded".format(args.rank)))

        del checkpoint
    
    #----------------------------------------------------
    #  PS-KD train & validation
    #----------------------------------------------------
    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.end_epoch):

        adjust_learning_rate(optimizer, epoch, args)
        
        alpha_t = args.alpha_T * ((epoch + 1) / args.end_epoch)
        alpha_t = max(0, alpha_t)

        train(None, None, None, criterion_student,
              criterion_teacher,optimizer, net, epoch, alpha_t, train_loader, args)

        #---------------------------------------------------
        #  Validation
        #---------------------------------------------------
        acc = val(criterion_CE, net, epoch, valid_loader, args)

        #---------------------------------------------------
        #  Save_dict for saving model
        #---------------------------------------------------
        save_dict = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc' : best_acc,
                    'accuracy' : acc,
                    'alpha_t' : alpha_t
                    }

        if acc > best_acc:
            best_acc = acc
            #save_on_master(save_dict,os.path.join(model_dir, 'checkpoint_best.pth'))
            
        #if args.saveckp_freq and (epoch+1) % args.saveckp_freq == 0:
        if epoch==1 or epoch==10 or epoch==50 or (epoch+1)%args.saveckp_freq == 0:
            save_on_master(save_dict,os.path.join(model_dir, f'checkpoint_{epoch:03}.pth'))
    return best_acc
            


#-------------------------------
# Train 
#------------------------------- 
def train(all_predictions,
          criterion_CE,
          criterion_CE_pskd,
          criterion_student,
          criterion_teacher,
          optimizer,
          net,
          epoch,
          alpha_t,
          train_loader,
          args):
    
    
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    train_losses = AverageMeter()
    
    correct = 0
    total = 0

    net.train()
    current_LR = get_learning_rate(optimizer)[0]

    for batch_idx, (inputs, targets, input_indices) in enumerate(train_loader):
        
        inputs = inputs.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
            
        #-----------------------------------
        # Self-KD or none
        #-----------------------------------                
        
        targets_numpy = targets.cpu().detach().numpy()
        identity_matrix = torch.eye(len(train_loader.dataset.classes)) 
        targets_one_hot = identity_matrix[targets_numpy]
            
        # student model
        # compute output
        embedding = net(inputs)
        detached_embedding = embedding.clone().detach()

        if args.use_student_loss:
           student_logits = net.student_head(embedding)
        
        else:
           student_logits = net.student_head(detached_embedding)


        if args.use_teacher_loss:
           teacher_logits = net.learnable_params(F.normalize(net.teacher_head(embedding)))

        else:
           teacher_logits = net.learnable_params(F.normalize(net.teacher_head(detached_embedding)))   
        
        
        loss_student = criterion_student(student_logits, targets_one_hot.cuda(),
                                        teacher_logits.clone().detach(), alpha_t)
        
        loss_teacher = criterion_teacher(teacher_logits, targets)

        loss = loss_student + loss_teacher

        if args.use_prior:
            prior_loss = net.prior_loss(args.data_type)
            loss += prior_loss
        
        train_losses.update(loss.item(), inputs.size(0))
        err1, err5 = accuracy(student_logits.data, targets, topk=(1, 5))
        train_top1.update(err1.item(), inputs.size(0))
        train_top5.update(err5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # after optimizer step, normalize the learnable parameters again
        with torch.no_grad():
            net.learnable_params.weight.div_(torch.norm(net.learnable_params.weight, dim=1, keepdim=True))

        _, predicted = torch.max(student_logits, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar(epoch,batch_idx, len(train_loader), args, 'lr: {:.1e} | alpha_t: {:.3f} | loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | correct/total({}/{})'.format(
            current_LR, alpha_t, train_losses.avg, train_top1.avg, train_top5.avg, correct, total))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [lr {:.1e}] [alpht_t {:.3f}] [train_loss {:.3f}] [train_top1_acc {:.3f}] [train_top5_acc {:.3f}] [correct/total {}/{}]'.format(
        epoch,
        current_LR,
        alpha_t,
        train_losses.avg,
        train_top1.avg,
        train_top5.avg,
        correct,
        total))
    

#-------------------------------          
# Validation
#------------------------------- 
def val(criterion_CE,
        net,
        epoch,
        val_loader,
        args):


    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_losses = AverageMeter()


    targets_list = []
    confidences = []

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):              
            
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
                
            #for ECE, AURC, EAURC
            targets_numpy = targets.cpu().numpy()
            targets_list.extend(targets_numpy.tolist())
                
            # model output
            student_logits = net.student_head(net(inputs))
            
            # for ECE, AURC, EAURC
            student_prob = F.softmax(student_logits, dim=1)
            student_prob = student_prob.cpu().numpy()
            for values_ in student_prob:
                confidences.append(values_.tolist())
                
            _, predicted = torch.max(student_logits, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loss = criterion_CE(student_logits, targets)
            val_losses.update(loss.item(), inputs.size(0))
            
            #Top1, Top5 Err
            err1, err5 = accuracy(student_logits.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))

            progress_bar(epoch, batch_idx, len(val_loader), args,'val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | correct/total({}/{})'.format(
                        val_losses.avg,
                        val_top1.avg,
                        val_top5.avg,
                        correct,
                        total))
            
    #if is_main_process():
    ece,aurc,eaurc = metric_ece_aurc_eaurc(confidences,
                                               targets_list,
                                               bin_size=0.1)

    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [val_loss {:.3f}] [val_top1_acc {:.3f}] [val_top5_acc {:.3f}] [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}] [correct/total {}/{}]'.format(
                    epoch,
                    val_losses.avg,
                    val_top1.avg,
                    val_top5.avg,
                    ece,
                    aurc,
                    eaurc,
                    correct,
                    total))


    return val_top1.avg



if __name__ == '__main__':
    main()
