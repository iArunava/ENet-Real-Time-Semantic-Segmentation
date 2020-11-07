import torch
import torch.nn as nn
from utils import *
from models.ENet import ENet
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def test(FLAGS):
    # Check if the pretrained model is available
    if not FLAGS.m.endswith('.pth'):
        raise RuntimeError('Unknown file passed. Must end with .pth')
    if FLAGS.image_path is None or not os.path.exists(FLAGS.image_path):
        raise RuntimeError('An image file path must be passed')
    
    h = FLAGS.resize_height
    w = FLAGS.resize_width

    checkpoint = torch.load(FLAGS.m,  map_location=FLAGS.cuda)
    
    # Assuming the dataset is camvid
    enet = ENet(FLAGS.num_classes)
    enet.load_state_dict(checkpoint['state_dict'])

    tmg_ = plt.imread(FLAGS.image_path)
    tmg_ = cv2.resize(tmg_, (h, w), cv2.INTER_NEAREST)
    tmg = torch.tensor(tmg_).unsqueeze(0).float()
    tmg = tmg.transpose(2, 3).transpose(1, 2)

    with torch.no_grad():
        out1 = enet(tmg.float()).squeeze(0)
    
    #smg_ = Image.open('/content/training/semantic/' + fname)
    #smg_ = cv2.resize(np.array(smg_), (512, 512), cv2.INTER_NEAREST)

    b_ = out1.data.max(0)[1].cpu().numpy()

    decoded_segmap = decode_segmap(b_)

    images = {
        0 : ['Input Image', tmg_],
        1 : ['Predicted Segmentation', b_],
    }

    show_images(images)
