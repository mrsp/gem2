#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * GEM2 - Gait-phase Estimation Module 2
 *
 * Copyright 2020-2021 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH)
 *	     nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
'''

from gem2 import GEM2
from gem2_tools import GEM2_tools
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import sys
import yaml
import os
from os import path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from sklearn import mixture
from sklearn.cluster import KMeans
# import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

from variationalAutoencoder import variationalAutoencoder
from autoencoder import autoencoder
from supervisedAutoencoder import supervisedAutoencoder
from supervisedVariationalAutoencoder import supervisedVariationalAutoencoder
from supervisedClassifier import supervisedClassifier


def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
	config = load_config(sys.argv[1])
	train_path = os.path.dirname(os.path.realpath(
	    __file__)) + "/" + config['gem2_train_path']
	val_path = os.path.dirname(os.path.realpath(
	    __file__)) + "/" + config['gem2_validation_path']
	robot = config['gem2_robot']


	#Load the Training and Validation Datasets
	gt = GEM2_tools(validation=config['gem2_validation'], gt_comparison=config['gem2_gt_comparison'],
	               gem2=config['gem2'], useLabels=config['useLabels'])
	gt.input_data(train_path, val_path)
	data_train = gt.data_train
	data_labels = gt.data_label
	data_val = gt.data_val
	data_val_labels = gt.data_val_label


	#Load the Pre-Trained GEM Model
	#gem_model = load_model("talos_sim_AE.h5",compile=False)
	
	gem_model = load_model("nao_SAE.h5",compile=False)
	#gem_model = load_model("talos_sim_SAE.h5",compile=False)
	#gem_model = load_model("talos_real_SAE.h5",compile=False)

	
	out_path = train_path

	if(config['gem2']):
		reduced_data_train = gem_model.predict(data_train)[1]
		leg_probabilities = gem_model.predict(data_train)[2]
	
		if(not path.exists(out_path + "/" + "RLeg_probabilities.txt")):
			fileObj = open(out_path+ "/" + "RLeg_probabilities.txt", "w")
			np.savetxt(fileObj, leg_probabilities[:,1])
			fileObj.close()
		if(not path.exists(out_path + "/" + "LLeg_probabilities.txt")):
			fileObj = open(out_path+ "/" + "LLeg_probabilities.txt", "w")
			np.savetxt(fileObj, leg_probabilities[:,0])
			fileObj.close()
	else:
		reduced_data_train = gem_model.predict(data_train)


		#if(gt.useLabels):
			#gt.plot_accelerations_LR(leg_probabilities, data_labels)
			#gt.plot_accelerations_LRD(g.leg_probabilities, data_labels,g.predicted_labels_train)
		
	print("Clustering with Gaussian Mixture Models")
	#gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=200, tol=5.0e-2, init_params = 'kmeans', n_init=10,warm_start=False,verbose=1)
	gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=200, tol=5.0e-2, init_params = 'kmeans', n_init=100,warm_start=False,verbose=1)
	#reduced_data_train[:,0] = reduced_data_train[:,0] + 1.40
	#reduced_data_train[:,1] = reduced_data_train[:,1] + 1.60

	gmm.fit(reduced_data_train)
	predicted_labels_train = gmm.predict(reduced_data_train)


	
 
	
	if(config['gem2_gt_comparison']):
		gt.genGroundTruthStatistics(reduced_data_train)
		gt.plot_results(reduced_data_train, gt.phase, gt.mean, gt.covariance, 'Ground-Truth Labels')
		# gt.plot_latent_space(g)
		cnf_matrix = confusion_matrix(gt.phase,  predicted_labels_train)
		np.set_printoptions(precision=2)
		class_names = ['RSS','DS','LSS']
		gt.plot_confusion_matrix(cnf_matrix, class_names, 'GMMs Confusion Matrix')
	gt.plot_results(reduced_data_train, predicted_labels_train, gmm.means_, gmm.covariances_, 'Clustering with Gaussian Mixture Models')
		
