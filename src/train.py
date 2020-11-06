#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * GEM2 - Gait-phase Estimation Module
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
from threading import Thread 

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)



if __name__ == "__main__":

	config = load_config(sys.argv[1])
	train_path =  os.path.dirname(os.path.realpath(__file__)) + "/" + config['gem2_train_path']
	val_path =  os.path.dirname(os.path.realpath(__file__)) + "/" + config['gem2_validation_path']
	robot = config['gem2_robot']
	gt = GEM2_tools(validation = config['gem2_validation'], gt_comparison=config['gem2_gt_comparison'], gem2=config['gem2'], useLabels = config['useLabels'])
	gt.input_data(train_path,val_path)


	g = GEM2()
	g.setFrames(config['gem2_lfoot_frame'], config['gem2_rfoot_frame'])
	g.setParams(config['gem2_dim'], config['gem2'], config['gem2_robot'], config['gem2_save'])
	data_train = gt.data_train
	data_labels = gt.data_label
	data_val = gt.data_val
	data_val_labels = gt.data_val_label



	g.fit(data_train, data_val,  config['gem2_dim_reduction'], config['gem2_clustering'], data_labels, data_val_labels)
	if(gt.useLabels):
		gt.plot_accelerations_LR(g.leg_probabilities, data_labels)
		#gt.plot_accelerations_LRD(g.leg_probabilities, data_labels,g.predicted_labels_train)



	if(config['gem2_plot_results']):
		if(config['gem2_gt_comparison']):
			gt.genGroundTruthStatistics(g.reduced_data_train)
			gt.plot_results(g.reduced_data_train, gt.phase, gt.mean, gt.covariance, 'Ground-Truth Labels')
			#gt.plot_latent_space(g)
			predicted_labels_train = g.predicted_labels_train
			cnf_matrix = confusion_matrix(gt.phase,  predicted_labels_train)
			np.set_printoptions(precision=2)
			class_names = ['DS','LSS','RSS']
			gt.plot_confusion_matrix(cnf_matrix, class_names, 'Confusion matrix')

		if(config['gem2_clustering'] == "kmeans"):
			gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.kmeans.cluster_centers_, None, 'Clustering with K-means')
		elif(config['gem2_clustering'] == "gmm"):
			gt.plot_results(g.reduced_data_train, g.predicted_labels_train, g.gmm.means_, g.gmm.covariances_, 'Clustering with Gaussian Mixture Models')
		else:
			print("Unsupported Result Plotting")

	
	if(config['gem2_save']):
		with open(robot + '_gem2.sav', 'wb') as file:
			pickle.dump(g, file)
		with open(robot + '_gem2_tools.sav', 'wb') as file:
			pickle.dump(gt, file)
	print('Training Finished')
	
	