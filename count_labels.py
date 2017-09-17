from __future__ import division
from test_tf_record import create_tf_record, iterate_files
import glob

def find_train_and_val_distribution():

    all_images = glob.glob('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/train/*.jpg')
    
    label_file_array = []
    with open('labels.csv') as f:
        for index, row in enumerate(f):
            if index >= 1:
                label_file_array.append(row)

    file_pointers = []
    labels = []

    iterate_files(label_file_array, all_images, file_pointers,labels, ',')

    invasive_count = 0
    total_count = 0

    for l in labels:
        if l == 1:
            invasive_count +=1
        total_count +=1

    print "train invasive count:", invasive_count/total_count

    all_images = glob.glob('/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/val/*.jpg')
    
    label_file_array = []
    with open('labels.csv') as f:
        for index, row in enumerate(f):
            if index >= 1:
                label_file_array.append(row)

    file_pointers = []
    labels = []

    iterate_files(label_file_array, all_images, file_pointers,labels, ',')
   
    invasive_count = 0
    total_count = 0

    for l in labels:
        if l == 1:
            invasive_count +=1
        total_count +=1

    print "val invasive count:", invasive_count/total_count




if __name__ == '__main__':
    find_train_and_val_distribution()