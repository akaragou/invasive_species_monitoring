from __future__ import division
import glob
import shutil
import random

def make_train_val():

    all_data_path = '/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/downloaded_data/'
    train_path = '/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/train/'
    val_path = '/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/data/val/'

    full_image_paths = glob.glob(all_data_path + '*.jpg')

    images = []

    for path in full_image_paths:
        img = path.split('/')[-1]
        images.append(img)


    num_images = range(len(images))
    indicies = random.sample(num_images, 500)

    for i in range(len(images)):
        print "copying image number:{0}".format(i+1)
        if i in indicies:
            try:
                shutil.copy(all_data_path+images[i], val_path+images[i])
            except OSError:
                print "continuing"
        else:
            try:
                shutil.copy(all_data_path+images[i], train_path+images[i])
            except OSError:
                print "continuing"






if __name__ == '__main__':
    make_train_val()