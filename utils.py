##############################################################
###############   utilities for the project ###############
##############################################################
# this scripts contains all useful code that we might want to use in the separate notebooks 
# that implements the different models

import os, glob, csv, random
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torchaudio
from torch.utils.data import Dataset

# preprocessing functions and evaluation models
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Main dataset class that we use to read the audio data and feed it into ou models
class DCaseDataset(Dataset):
    """
    Dataloader for DCase dataset
    Structure of the class is taken from:
    https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/audio_classifier_tutorial.ipynb
    """

    labelind2name = {
        0: "airport",
        1: "bus",
        2: "metro",
        3: "metrostation",
        4: "park",
        5: "publicsquare",
        6: "shoppingmall",
        7: "streetpedestrian",
        8: "streettraffic",
        9: "tram",
    }
    name2labelind = {
        "airport": 0,
        "bus": 1,
        "metro": 2,
        "metrostation": 3,
        "park": 4,
        "publicsquare": 5,
        "shoppingmall": 6,
        "streetpedestrian": 7,
        "streettraffic": 8,
        "tram": 9,
    }

    def __init__(self, root_dir, split=None):
        """

        :param root_dir:
        """

        self.root_dir = root_dir

        self.split=split

        # Lists of file names and labels
        self.filepaths, self.labels = self.load_csv(os.path.join(self.root_dir,"data.csv"))
        
        
        """ NO DEVICE INFO
        # Device for each audio file
        self.devices = {}
        for i in range(0, len(metaData)):
            self.devices[metaData.iloc[i, 0]] = metaData.iloc[i, 3]
        """
        
        # Transform class name to index
        self.labels = [self.name2labelind[name] for name in self.labels]

    def load_csv(self, filename):
        """ 
        returns the list of filenames and their labels 
        They are stored in a csv file to speed up the file search process
        if no csv exists, it creates it
        """


        # if no .csv, create; else load    
        if not os.path.exists(filename):

            sounds = []
            for name in self.name2labelind.keys():
                # 'data\label\sound.wav'
                print(f'Searching for {name} in {os.path.join(self.root_dir, "audio/" + name, "*.wav")}')
                sounds += glob.glob(os.path.join(self.root_dir, "audio/" + name, '*.wav'))          
            
            print(len(sounds), sounds)   

            random.shuffle(sounds)
            
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                for snd in sounds:
                    name = snd.split(os.sep)[-2].split('_')[-1]
                    writer.writerow([snd, name])
                print(f"written into csv file: {filename}")        
        
        
        # read from csv file
        sounds, labels = [], []
        with open(filename) as f:
            reader = csv.reader(f)

            for row in reader:
                snd, label = row                
                sounds.append(snd)
                labels.append(label)        

        assert len(sounds) == len(labels)        
        return sounds, labels


    def __getitem__(self, index):
        """
        :param index:
        :return:
        """

        # Load data
        filepath = self.filepaths[index]
        sound, sfreq = torchaudio.load(filepath)
        assert sound.shape[0] == 1, "Expected mono channel"
        sound = torch.mean(sound, dim=0)
        assert sfreq == 44100, "Expected sampling rate of 44.1 kHz"

        # Remove last samples if longer than expected
        if sound.shape[-1] >= 441000:
            sound = sound[:441000]

        if self.split == "test":
            return sound, 255, self.filepaths[index], "unknown"
        else:
            return (
                sound,
                self.labels[index],
                self.filepaths[index],
            )

    def __len__(self):
        return len(self.filepaths)

# gives the number of parameters of a given torch model 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model,test_loader):
    return

#Plot fancy confusion matrix 
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax