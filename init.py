import numpy as np
import argparse
from train import *
from test import *

color_map = {
    'unlabeled'     : (  0,  0,  0),
    'dynamic'       : (111, 74,  0),
    'ground'        : ( 81,  0, 81),
    'road'          : (128, 64,128),
    'sidewalk'      : (244, 35,232),
    'parking'       : (250,170,160),
    'rail track'    : (230,150,140),
    'building'      : ( 70, 70, 70),
    'wall'          : (102,102,156),
    'fence'         : (190,153,153),
    'guard rail'    : (180,165,180),
    'bridge'        : (150,100,100),
    'tunnel'        : (150,120, 90),
    'pole'          : (153,153,153),
    'traffic light' : (250,170, 30),
    'traffic sign'  : (220,220,  0),
    'vegetation'    : (107,142, 35),
    'terrain'       : (152,251,152),
    'sky'           : ( 70,130,180),
    'person'        : (220, 20, 60),
    'rider'         : (255,  0,  0),
    'car'           : (  0,  0,142),
    'truck'         : (  0,  0, 70),
    'bus'           : (  0, 60,100),
    'caravan'       : (  0,  0, 90),
    'trailer'       : (  0,  0,110),
    'train'         : (  0, 80,100),
    'motorcycle'    : (  0,  0,230),
    'bicycle'       : (119, 11, 32)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        type=str,
                        default='./datasets/CamVid/ckpt-camvid-enet.pth',
                        help='The path to the pretrained enet model')

    parser.add_argument('-i', '--image-path',
                        type=str,
                        help='The path to the image to perform semantic segmentation')

    parser.add_argument('-rh', '--resize-height',
                        type=int,
                        default=512,
                        help='The height for the resized image')

    parser.add_argument('-rw', '--resize-width',
                        type=int,
                        default=512,
                        help='The width for the resized image')

    parser.add_argument('-lr', '--learning-rate',
                        type=float,
                        default=5e-4,
                        help='The learning rate')

    parser.add_argument('-bs', '--batch-size',
                        type=int,
                        default=10,
                        help='The batch size')

    parser.add_argument('-wd', '--weight-decay',
                        type=float,
                        default=2e-4,
                        help='The weight decay')

    parser.add_argument('-c', '--constant',
                        type=float,
                        default=1.02,
                        help='The constant used for calculating the class weights')

    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=102,
                        help='The number of epochs')

    parser.add_argument('-nc', '--num-classes',
                        type=int,
                        default=12,
                        help='Number of unique classes')

    parser.add_argument('-se', '--save-every',
                        type=int,
                        default=10,
                        help='The number of epochs after which to save a model')

    parser.add_argument('-iptr', '--input-path-train',
                        type=str,
                        default='./datasets/CamVid/train/',
                        help='The path to the input dataset')

    parser.add_argument('-lptr', '--label-path-train',
                        type=str,
                        default='./datasets/CamVid/trainannot/',
                        help='The path to the label dataset')

    parser.add_argument('-ipv', '--input-path-val',
                        type=str,
                        default='./datasets/CamVid/val/',
                        help='The path to the input dataset')

    parser.add_argument('-lpv', '--label-path-val',
                        type=str,
                        default='./datasets/CamVid/valannot/',
                        help='The path to the label dataset')

    parser.add_argument('-iptt', '--input-path-test',
                        type=str,
                        default='./datasets/CamVid/test/',
                        help='The path to the input dataset')

    parser.add_argument('-lptt', '--label-path-test',
                        type=str,
                        default='./datasets/CamVid/testannot/',
                        help='The path to the label dataset')

    parser.add_argument('-pe', '--print-every',
                        type=int,
                        default=1,
                        help='The number of epochs after which to print the training loss')

    parser.add_argument('-ee', '--eval-every',
                        type=int,
                        default=10,
                        help='The number of epochs after which to print the validation loss')

    parser.add_argument('--cuda',
                        type=bool,
                        default=False,
                        help='Whether to use cuda or not')

    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='Whether to train or test')
    
    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.cuda = torch.device('cuda:0' if torch.cuda.is_available() and FLAGS.cuda \
                               else 'cpu')

    if FLAGS.mode.lower() == 'train':
        train(FLAGS)
    elif FLAGS.mode.lower() == 'test':
        test(FLAGS)
    else:
        raise RuntimeError('Unknown mode passed. \n Mode passed should be either \
                            of "train" or "test"')
