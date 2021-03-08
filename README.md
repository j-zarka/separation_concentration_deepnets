### Code implementation of paper: Separation and Concentration in Deep Networks
This repository contains the code to reproduce experiments in the paper: [Separation and Concentration in Deep Networks](https://openreview.net/forum?id=8HhkbjrWLdE) 
currently accepted at ICLR 2021 conference.

### Requirements
Our code is designed to run on GPU using [PyTorch](https://pytorch.org/) framework, while scattering transforms are computed using the [Kymatio software package](https://github.com/kymatio/kymatio)
which supports torch and scikit-cuda ('skcuda') backends (skcuda being faster).
In order to run our experiments you will need the following packages: 
- For PyTorch: torch, torchvision, tensorboard
- For skcuda: scikit-cuda, cupy

and a multi-GPU version of Kymatio.

You can install the PyTorch and skcuda packages by:

`pip install torch torchvision tensorboard scikit-cuda cupy`

For a multi-GPU version of Kymatio, you can use the _multigpu_ branch of our Kymatio fork 
https://github.com/j-zarka/kymatio

To install this branch:

```
git clone -b multigpu https://github.com/j-zarka/kymatio.git
cd kymatio
pip install -r requirements.txt
pip install .
```

### Phase scattering
The complex wavelet frame described in section 3.1 of the paper is implemented 
in a separate torch module phase_scattering2d_torch. The scattering tree which 
averages over phases rectified wavelet coefficients and thus
approximatively computes a complex modulus is directly computed with Kymatio.

### ImageNet
Download ImageNet dataset from http://www.image-net.org/challenges/LSVRC/2012/downloads (registration required).
Then move validation images to labeled subfolders, using [the PyTorch shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

Model can optionally be trained on a subset of ImageNet classes using the --nb-classes option (with --nb-classes < 1000).
In this case, selected classes' indices are either randomly chosen or determined in a file whose path is specified using the --class-indices
option. Examples of such file for 10, 20 and 100 classes are provided in the utils_sampling folder.

### Setup
Results in the last version of the paper were produced using torch 1.7.0, torchvision 0.8.0 and cuda 11.1
 
### Usage
To train a model, run main.py with the desired model architecture and dataset, and the below options.

To reproduce the paper's experiments, run the following commands:

For classification on MNIST:
- Using a 2-layer network:

```
python main.py  --dataset mnist -a analysis -j 10 --frame-kernel-size 14 --frame-width 2048 --frame-stride 7 
--epochs 300 --batch-size 128 --lr 0.01 --learning-rate-adjust-frequency 70 --non-linearity [relu/softshrink/absolute]
--dir path/to/dir
```
- Using scattering tree:
```
python main.py  --dataset mnist -a scatnet -j 10 --scattering-J 3 --P-proj-size 128 --epochs 300 --batch-size 128 
--lr 0.01 --learning-rate-adjust-frequency 70 --dir path/to/dir 
```     

For classification on CIFAR10:
- Using a 2-layer network:

```
python main.py  --dataset cifar10 -a analysis -j 10 --frame-kernel-size 8 --frame-width 8192 --frame-stride 4 
--epochs 300 --batch-size 128 --lr 0.01 --learning-rate-adjust-frequency 70 --non-linearity [relu/softshrink/absolute]
--dir path/to/dir
```

- Using scattering tree:
```
python main.py  --dataset cifar10 -a scatnet -j 10 --scattering-J 3 --P-proj-size 128 --epochs 300 --batch-size 128 
--lr 0.01 --learning-rate-adjust-frequency 70 --dir path/to/dir 
```  

- Using scattering with learned projections:
```
python main.py  --dataset cifar10 -a scatnetblock -j 10 --n-blocks 4 --P-proj-size 64 128 256 512 --epochs 300 
--batch-size 128 --lr 0.01 --learning-rate-adjust-frequency 70 --dir path/to/dir  
``` 

- Using scattering with learned projections and concentration:
```
 python main.py  --dataset cifar10 -a scatnetblockanalysis -j 10 --n-blocks 4  
--P-proj-size 64 128 256 512 --non-linearity [relu/softshrink] --frame-width 1024 2048 4096 8192
--epochs 300 --batch-size 128 --lr 0.01 --learning-rate-adjust-frequency 70 --dir path/to/dir
```

For classification on ImageNet:

- Using scattering tree:
```
python main.py  --dataset imagenet -a scatnet -j 10 --scattering-J 4 --P-proj-size 256 --epochs 200 --batch-size 128 
--lr 0.01 --learning-rate-adjust-frequency 60 --avg-ker-size 5 --dir path/to/dir --data path/to/imagenet
```  

- Using scattering with learned projections:
```
python main.py  --dataset imagenet -a scatnetblock -j 10 --n-blocks 6 --P-proj-size 64 128 256 512 512 512
--epochs 200 --batch-size 128 --lr 0.01 --learning-rate-adjust-frequency 60 --dir path/to/dir --data path/to/imagenet
``` 

- Using scattering with learned projections and concentration:
```
python main.py  --dataset imagenet -a scatnetblockanalysis -j 10 --n-blocks 6 --epochs 200 
--P-proj-size 64 128 256 512 512 512 --non-linearity [relu/softshrink] --frame-width 512 1024 1024 2048 2048 2048
--batch-size 128 --lr 0.01 --learning-rate-adjust-frequency 60 --dir path/to/dir --data path/to/imagenet
```

For more details, please see below usage.

```
usage: main.py [-h] [--data DATA]
               [--dataset {mnist,cifar10,cifar100,imagenet}]
               [-a {analysis,scatnet,scatnetblock,scatnetblockanalysis}]
               [-j WORKERS] [--epochs EPOCHS] [--start-epoch START_EPOCH]
               [-b BATCH_SIZE] [--lr LR] [--momentum MOMENTUM]
               [--wd WEIGHT_DECAY] [-p PRINT_FREQ] [--resume RESUME] [-e]
               [--seed SEED]
               [--learning-rate-adjust-frequency LEARNING_RATE_ADJUST_FREQUENCY]
               [--dir DIR] [--pars-beta PARS_BETA] [--n-blocks N_BLOCKS]
               [--scat-angles SCAT_ANGLES [SCAT_ANGLES ...]]
               [--backend BACKEND] [--scattering-J SCATTERING_J]
               [--P-proj-size P_PROJ_SIZE [P_PROJ_SIZE ...]]
               [--P-kernel-size P_KERNEL_SIZE [P_KERNEL_SIZE ...]]
               [--non-linearity {absolute,softshrink,relu}] [--zero-bias]
               [--frame-width FRAME_WIDTH [FRAME_WIDTH ...]]
               [--frame-kernel-size FRAME_KERNEL_SIZE [FRAME_KERNEL_SIZE ...]]
               [--frame-stride FRAME_STRIDE [FRAME_STRIDE ...]]
               [--classifier-type {fc,mlp}] [--avg-ker-size AVG_KER_SIZE]
               [--nb-hidden-units NB_HIDDEN_UNITS]
               [--dropout-p-mlp DROPOUT_P_MLP] [--nb-l-mlp NB_L_MLP]
               [--nb-classes NB_CLASSES] [--class-indices CLASS_INDICES]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to dataset
  --dataset {mnist,cifar10,cifar100,imagenet}
                        dataset to train on (default: imagenet)
  -a {analysis,scatnet,scatnetblock,scatnetblockanalysis}, --arch {analysis,scatnet,scatnetblock,scatnetblockanalysis}
                        model architecture (default: scatnetblockanalysis)
  -j WORKERS, --workers WORKERS
                        number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 150)
  --start-epoch START_EPOCH
                        manual epoch number (useful on restarts)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.01)
  --momentum MOMENTUM   momentum (default: 0.9)
  --wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight decay (default: 1e-4)
  -p PRINT_FREQ, --print-freq PRINT_FREQ
                        print frequency (default: 100)
  --resume RESUME       path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --seed SEED           seed for initializing training
  --learning-rate-adjust-frequency LEARNING_RATE_ADJUST_FREQUENCY
                        number of epoch after which learning rate is decayed
                        by 10 (default: 30)
  --dir DIR             directory for training logs and checkpoints
  --pars-beta PARS_BETA
                        learning rate for pars reg (default: 0.0005)
  --n-blocks N_BLOCKS   number of blocks in the pipeline (default: 4)
  --scat-angles SCAT_ANGLES [SCAT_ANGLES ...]
                        number of orientations for wavelet frame(s) (default:
                        8)
  --backend BACKEND     scattering backend
  --scattering-J SCATTERING_J
                        maximum scale for the scattering transform - for
                        scatnet arch only (default: 4)
  --P-proj-size P_PROJ_SIZE [P_PROJ_SIZE ...]
                        output dimension of the linear projection(s) P(s)
                        (default: 256)
  --P-kernel-size P_KERNEL_SIZE [P_KERNEL_SIZE ...]
                        kernel size of P(s) (default: 1)
  --non-linearity {absolute,softshrink,relu}
                        non linearity for analysis (default: relu)
  --zero-bias           force zero bias for ReLU
  --frame-width FRAME_WIDTH [FRAME_WIDTH ...]
                        size(s) of tight frame(s) (default: 2048)
  --frame-kernel-size FRAME_KERNEL_SIZE [FRAME_KERNEL_SIZE ...]
                        kernel size of frame(s) (default: 1)
  --frame-stride FRAME_STRIDE [FRAME_STRIDE ...]
                        stride of frame(s) (default: 1)
  --classifier-type {fc,mlp}
                        classifier type (default: fc)
  --avg-ker-size AVG_KER_SIZE
                        size of averaging kernel (default: 1)
  --nb-hidden-units NB_HIDDEN_UNITS
                        number of hidden units for mlp classifier (default:
                        2048)
  --dropout-p-mlp DROPOUT_P_MLP
                        dropout probability in mlp (default: 0.3)
  --nb-l-mlp NB_L_MLP   number of hidden layers in mlp (default: 2)
  --nb-classes NB_CLASSES
                        ImageNet only - number of classes randomly chosen used
                        for training and validation (default: 1000 = whole
                        train/val dataset)
  --class-indices CLASS_INDICES
                        ImageNet only - numpy array of indices used in case
                        nb-classes < 1000
  ```

 