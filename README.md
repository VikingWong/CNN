# Convolutional neural network
This is a Theano implementation of a convolutional neural network. It is possible to configure all the settings, including
number of convolutional layers. The system also include hooks for a web interface, for monitoring experiments. A web interface
can be used to stop a running job, and other things. The purpose of this system is to create a road extraction system.

## Features
- Nesterov momentum and RMSProp
- Easy configuration
- Dataset loader which generate examples from random subsets of an rotated image
- Hooks for web gui interface

## Dependencies
* [Theano](http://deeplearning.net/software/theano/)
* [Numpy](http://www.numpy.org/)
* [Python 2.7](https://www.python.org/)
* [Unirest](http://unirest.io/python.html)

## Installation
### Ubuntu
**Preliminary guide. Check if this works for ubuntu**

1. Install all dependencies

  ```bash
  $ sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy  
  ```
  
2. Install Theano

  ```bash
  $ sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git  
  ```
  
3. Download Cuda 7 toolkit

  ```bash
  $ sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb  
  ```
  
4. Depackage Cuda

  ```bash
  $ sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb  
  ```

5. Install the cuda driver

  ```bash
  $ sudo apt-get update

  $ sudo apt-get install -y cuda   
  ```
  
6. Update the path to include cuda nvcc and ld_library_path, and then reboot.

  ```bash
  $ echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc  
  ```
  
7. Create a theano config file

  ```bash
  $ echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc    
  ```
  
### Windows
**Preliminary guide. Check if this works for Windows**

1. Install Visual Studio 2013 Community Edition
2. Download CUDA 7 toolkit
3. Install Anaconda Python 27
4. In Windows Command Prompt type

  ```bash
  $ conda install mingw libpython
  ```
5. Use git to download Theano from GitHub
6. In the theano folder type:

  ```bash
  $ python setup.py develop
  ```
7. Create a .theanorc file in your home directory and add
  ```txt
    [global]
    floatX = float32
    device = gpu

    [nvcc]
    flags=-LC:\Anaconda\libs
    compiler_bindir=C:\Program Files (x86)\Microsoft Visual Studio 13.0\VC\bin
  ```
  Make sure the paths for anaconda and visual studio are correct.

##Profile
In your .theanorc file, change or enter profile=True.
Run system by command CUDA_LAUNCH_BLOCKING=1 python cnn.py
