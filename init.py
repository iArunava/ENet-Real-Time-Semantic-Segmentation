import numpy as np
import argparse
from train import *
from test import *

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
                        default=102,
                        help='The number of epochs')

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
