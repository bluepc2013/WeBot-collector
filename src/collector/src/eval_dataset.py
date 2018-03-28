#!/usr/bin/python
# -*- coding: utf-8 -*-
"""This program is used to read a dataset, and show its image and label.
# Author: zhangping
# Create at : 2018.1.3
"""

import random
import pdb
import matplotlib.pyplot as plt
from image_dataset import read_dataset

#
DATASET_DIR = '/home/blue/lab/robot/omnidirectional_vehicle/neural_planner_dataset_V2.0/tmp/train_dir/'
DATASET_IMAGES_FILE = 'train-dataset-images.gz'
DATASET_LABELS_FILE = 'train-dataset-labels.gz'


def show_mat(mat, label=''):
    """show a 2-D matrix."""
    if mat.ndim != 2:
        print "Error at show_mat:  @mat must be a 2-D matrix."

    plt.imshow(mat, cmap=plt.cm.gray)
    plt.title(label)
    plt.show()
    plt.clf()

# main
if __name__ == '__main__':
    # Load Image
    dataset = read_dataset(DATASET_DIR, DATASET_IMAGES_FILE\
                            , DATASET_LABELS_FILE)

    # show some of images.
    img_num = dataset.images.shape[0]
    show_img_num = 100
    cell = img_num/show_img_num
    if cell < 1:
        cell = 1
        show_img_num = img_num-1

    img_rows = dataset.image_rows
    img_cols = dataset.image_cols
    print 'img_num:%d\nimg_rows:%d\nimg_cols:%d'%(img_num, img_rows, img_cols)
    for i in xrange(0, show_img_num):
        index = random.randint(i*cell, (i+1)*cell)
        img = dataset.images[index]
        lab = dataset.labels[index]
        img = img.reshape(img_rows, img_cols)
        show_mat(img, lab)
