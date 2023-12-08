# DeepLearning_Project_AcousticClassificationDCase
Repository of course project based on DCase 2021 Challenge

Authors: Barberteguy Victor


Based on the [following repository](https://github.com/dataflowr/Project-Acoustic-Scene-Classification-DCASE.git) 

This is a project done during the Deep Learning class by Prof. Umut Simsekli. It explores Deep Learning applied to audio scene classification, with CNN-based architectures (Resnet, CNN6) and several tricks implemented [here](https://arxiv.org/abs/2105.13734) (the github page is below). I also investigate the AudioCLIP mulitmodal founddation model for such audio classification task.

In this repository, you will find all you need to download and run the project (**except the dataset**, that you will have to download).

There are several notebooks:
- *CNNs.ipynb* where we train the CNN models
- *CLIP.ipynb$ where we classify our audio recordings thanks to CLIP, and visualize the latent space
- *ConvolutionDifferences.ipynb* where we investigate the differences between classical and separable convolution. You can also visualize what a Log Mel Spectrogram looks like.


-----------------
## Separable convolutions and test-time augmentations for low-complexity and calibrated acoustic scene classification (DCASE21 Challenge)

[**SEPARABLE CONVOLUTIONS AND TEST-TIME AUGMENTATIONS FOR LOW-COMPLEXITY AND CALIBRATED ACOUSTIC SCENE CLASSIFICATION**]() 

[*Gilles Puy*](https://sites.google.com/site/puygilles/home),
[*Himalaya Jain*](https://himalayajain.github.io/),
[*Andrei Bursuc*](https://abursuc.github.io/)  
*valeo.ai, Paris, France*

This repo contains the code to reproduce the results of the systems we submitted to the Task1a of the DCASE21 challenge. 
Please refer to [link1](http://dcase.community/challenge2021/task-acoustic-scene-classification#subtask-a) and 
[link2](https://arxiv.org/abs/2105.13734) for more information about the challenge.


If you find this code useful, please cite our [technical report]():
```
@techreport{vai21dcase,
  title={Separable convolutions and test-time augmentations for low-complexity and calibrated acoustic scene classification},
  author={Puy, Gilles and Jain, Himalaya and Bursuc, Andrei},
  institution={{DCASE2021 Challenge}},
  year={2021},
}
```
-----------------



## Installation
 
### DCASE21 Datasets
If not already done, please download the development and evaluation datasets from
[here](http://dcase.community/challenge2021/task-acoustic-scene-classification#download). 

We use a smaller size of the dataset in order to spare compuation time and power: we kept only 1000 audio samples for each class. The dataset is available [here](https://filesender.renater.fr/?s=download&token=7da7d036-7eb4-4323-8ea1-905aa581fa89)


### code

 * clone this repository

```
git clone https://github.com/JaybeeH32/MAP583_Project_AcousticClassificationDCase.git
```

 * copy database  in folder /data 


### Requirements 
* Python == 3.7
* CUDA >= 10.2
```bash
$ pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install tqdm scikit-learn tensorboard pandas pyaml torchlibrosa
$ apt install -y libsndfile1
```


you can use a conda environment:

```

```
