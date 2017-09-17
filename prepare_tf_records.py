#!/usr/bin/env python
from __future__ import division
import glob
import numpy as np
import re
from tf_record import create_tf_record, iterate_files


def build_train_tfrecords():
    """
    builds train tensorflow records (binary files) 
    Input: None
    Ouptut: None
    """
    all_images = glob.glob('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/train/*.jpg') # full filepaths to the images
    
    label_file_array = []
    with open('labels.csv') as f:
        for index, row in enumerate(f):
            if index >= 1:
                label_file_array.append(row)

    file_pointers = []
    labels = []

    print "Creating training pointers and labels..."
    iterate_files(label_file_array, all_images, file_pointers,labels, ',')
    print  "Done creating training pointers and labels!"
   
    create_tf_record('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/tfrecords/train.tfrecords',
     file_pointers, labels, resize=True)


def build_val_tfrecords():
    """
    builds val tensorflow records (binary files) 
    Input: None
    Ouptut: None
    """
    all_images = glob.glob('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/val/*.jpg') # full filepaths to the images
    
    label_file_array = []
    with open('labels.csv') as f:
        for index, row in enumerate(f):
            if index >= 1:
                label_file_array.append(row)

    file_pointers = []
    labels = []

    print "Creating val pointers and labels..."
    iterate_files(label_file_array, all_images, file_pointers,labels, ',')
    print  "Done creating val pointers and labels!"
   
    create_tf_record('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/tfrecords/val.tfrecords', 
        file_pointers, labels, resize=True)


def build_test_tfrecords():
    """
    builds test tensorflow records (binary files) 
    Input: None
    Ouptut: None
    """
    all_images = glob.glob('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/test/*.jpg') # full filepaths to the images

    create_tf_record('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/tfrecords/test.tfrecords',
    all_images, None, is_grayscale=False, resize=True)



if __name__ == '__main__':
    build_train_tfrecords()
    build_val_tfrecords()
    build_test_tfrecords()
    