# gem2
Gait-phase Estimation Module 2 (GEM2) for Humanoid Robot Walking. The code is open-source (BSD License). Please note that this work is an on-going research and thus some parts are not fully developed yet. Furthermore, the code will be subject to changes in the future which could include greater re-factoring.

GEM2 is a semisupervised learning framework which employs a 2D latent space and Gaussian Mixture Models (GMMs) to facilitate accurate prediction/classification of the gait phase during locomotion.

Nevertheless, GEM can be used for real-time gait phase estimation without training based on the contact wrenches and optionally kinematics. The latter functionality facilitates the case where not enough training data can be obtained. A C++ implementation is available as part of the State Estimation for RObot Walking (SEROW) framework at https://github.com/mrsp/serow 



Papers: 

### New feature: GEM State Publisher (https://github.com/mrsp/gem_state_publisher) 
A ROS - C/C++ package for gathering  all necessary data for GEM in real-time.

## Training
Solely proprioceptive sensing is utilized in training, namely joint encoder, F/T, and IMU.


## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
* Ubuntu 16.04 and later
* ROS kinetic and later
* Sklearn 
* Keras 2.2.4
* tensorflow 
* tested on python3 (3.6.9) and python (2.7.17)

## Installing
* pip install tensorflow
* pip install --no-dependencies keras==2.2.4
* pip install sklearn
* git clone https://github.com/mrsp/gem.git
* catkin_make
* If you are using catkin tools run: catkin build  


### Run in real-time to infer the gait-phase:
* configure appropriately the config yaml file (in config folder) with the corresponding topics 
* roslaunch gem gem_ros.launch
