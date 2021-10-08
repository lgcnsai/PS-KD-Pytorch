#--------------
#CNN-architecture
#--------------
from models import *

#--------------
#util
#--------------
from utils.color import Colorer


C = Colorer.instance()

def get_network(args):
    ################################
    #Declare instance for Clasifier#
    ################################

    if args.data_type == 'cifar100':
        if args.classifier_type == 'PyramidNet':
            net = PyramidNet(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
        elif args.classifier_type == 'PyramidNet_SD':
            net = PyramidNet_ShakeDrop(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
        elif args.classifier_type == 'ResNet18':
            net = CIFAR_ResNet18_preActBasic(num_classes=100)
        elif args.classifier_type == 'ResNet101':
            net = CIFAR_ResNet101_Bottle(num_classes=100)
        elif args.classifier_type == 'DenseNet121':
            net = CIFAR_DenseNet121(num_classes=100, bias=True)
        elif args.classifier_type == 'ResNeXt':
            net = CifarResNeXt(cardinality=8, depth=29, nlabels=100, base_width=64, widen_factor=4)

    if args.data_type == 'imagenet':
        if args.classifier_type == 'ResNet152':
            net = ResNet(dataset = 'imagenet', depth=152, num_classes=1000, bottleneck=True)                
 
    print(C.underline(C.yellow("[Info] Building model: {}".format(args.classifier_type))))


    return net