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
### Windows
####TODO: Check if this works on windows!
1. Install Visual Studio 2013 Community Edition
2. Download CUDA 7 toolkit
3. Install Anaconda Python 27
4. In Windows Command Prompt type:

  ```bash
  $ conda install mingw libpython
  ```
5. Use git to download Theano from GitHub
6. In the theano folder type:

  ```bash
  $ python setup.py develop
  ```
7. Create a .theanorc.txt file in your home directory and add:
  ```txt
    [global]
    floatX = float32
    device = gpu

    [nvcc]
    flags=-LC:\Anaconda\libs
    compiler_bindir=C:\Program Files (x86)\Microsoft Visual Studio 13.0\VC\bin
  ```
  Make sure the paths for anaconda and visual studio are correct.
