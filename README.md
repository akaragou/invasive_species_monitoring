# Invasive Species Monitoring Kaggle Competition

Finetuning fully connected layers of Vgg16 for the binary classification task of whether there is an invasive species present or not. 

<a href="https://www.kaggle.com/c/invasive-species-monitoring">Offical Competition Overview</a>

---
- count\_labels.py - counts the number of invasive species in train data set and test data set
- create\_train\_val.py - splits the data into train data set and val data set
- embedding\_visualization.py - creates sprite images, meta data and embedding config to be used with tensorboard
- extract\_features\_vgg.py - stores fc7 features and image filepaths for val image set to be used with embedding\_visualization.py
- labels.csv - labels for the train and val data
- prepare\_tf\_records.py - creates tfrecords for train, val and test
- test\_vgg.py - calculates accuracy for validation set or creates a csv with predictions for test set
- tf\_record.py - contains methods to create and decode tf records
- train\_vgg.py -  finetunes vgg16 batchnormalized and full connected layers and stores optimal model checkpoints
- vgg.py - tensorflow slim model definitions of vgg network architectures
- vgg\_config.py - stores values for filepaths and model parameters
