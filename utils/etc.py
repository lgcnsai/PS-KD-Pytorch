'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
'''

import torch
from torch import nn
import torch.distributed as dist


import os, logging
import sys
import time

#--------------
#util
#--------------
from utils.color import Colorer, ColorerContext

C = Colorer.instance()

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time


def progress_bar(epoch,current, total, args, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    C.yellow("Epoch: [{}]".format(epoch))
    sys.stdout.write(C.cyan2("Epoch: [{}/{}]".format(epoch, (args.end_epoch - 1))))
    sys.stdout.write(C.cyan2(' ['))
    for i in range(cur_len):
        sys.stdout.write(C.cyan2('='))
    sys.stdout.write(C.cyan2('>'))
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(C.cyan2(']'))

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(C.cyan2(msg))
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(C.cyan2(' %d/%d ' % (current+1, total)))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

        
def paser_config_save(args,PATH):
    import json
    with open(PATH+'/'+'config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

        
def set_logging_defaults(logdir, args):
    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])
    # log cmdline argumetns
    logger = logging.getLogger('main')
    if is_main_process():
        logger.info(args)