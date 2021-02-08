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

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml
import os
import pickle
from gem2 import GEM2

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
	config = load_config(sys.argv[1])
	robot = config['gem2_robot']

	model_path = config['gem2_path_to_models']
	gt = pickle.load(open(model_path + '/' + robot + '_gem2_tools.sav', 'rb'))


	g = GEM2()
	g.setMethods(config['gem2_dim_reduction'], config['gem2_clustering'])
	g.setFrames(config['gem2_lfoot_frame'], config['gem2_rfoot_frame'])
	g.setParams(config['gem2_dim'], config['gem2'], config['gem2_robot'], True, model_path)

	#Load the Pre-Trained GEM Model
	
	data_train = gt.data_train
	data_labels = gt.data_label
	data_val = gt.data_val
	data_val_labels = gt.data_val_label

	#Get the latent-data of GEM2
	predicted_labels,  reduced_data, leg_probabilities =  g.predict_dataset(data_train)
	'''	
	if(config['gem2']):

		if(not os.path.exists(out_path + "/" + "RLeg_probabilities.txt")):
			fileObj = open(out_path+ "/" + "RLeg_probabilities.txt", "w")
			np.savetxt(fileObj, leg_probabilities[:,1])
			fileObj.close()
		if(not os.path.exists(out_path + "/" + "LLeg_probabilities.txt")):
			fileObj = open(out_path+ "/" + "LLeg_probabilities.txt", "w")
			np.savetxt(fileObj, leg_probabilities[:,0])
			fileObj.close()
	'''
	if(config['gem2_gt_comparison']):
		gt.genGroundTruthStatistics(reduced_data)
		gt.plot_results(reduced_data, gt.phase, gt.mean, gt.covariance, 'Ground-Truth Labels')
		# gt.plot_latent_space(g)
		cnf_matrix = confusion_matrix(gt.phase,  predicted_labels)
		np.set_printoptions(precision=2)
		class_names = ['RSS','DS','LSS']
		gt.plot_confusion_matrix(cnf_matrix, class_names, 'GMMs Confusion Matrix')

		
	if(config['gem2_clustering'] == "kmeans"):
		gt.plot_results(reduced_data, predicted_labels, g.kmeans.cluster_centers_, None, 'Clustering with K-means')
	elif(config['gem2_clustering'] == "gmm"):
		gt.plot_results(reduced_data, predicted_labels, g.gmm.means_, g.gmm.covariances_, 'Clustering with Gaussian Mixture Models')
	else:
		print("Unsupported Result Plotting")