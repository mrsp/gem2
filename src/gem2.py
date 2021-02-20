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
 *	 nor the names of its contributors may be used to endorse or promote products derived from
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


from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn import mixture
from sklearn.cluster import KMeans
from autoencoder import autoencoder
from supervisedAutoencoder import supervisedAutoencoder
#from variationalAutoencoder import variationalAutoencoder
#from supervisedVariationalAutoencoder import supervisedVariationalAutoencoder
from supervisedClassifier import supervisedClassifier
import pickle as pickle
from tensorflow.keras.models import load_model
import os
from math import exp
class GEM2():
    def __init__(self):
        self.firstrun = True
	
    def setParams(self, dim_, gem2, robot_, load_model_ = False, out_path = "."):
        self.latent_dim = dim_
        self.gem2 = gem2
        self.robot = robot_
        if(self.gem2):
            self.input_dim = 21
        else:
            self.input_dim = 15

        self.intermidiate_dim = 10


        if(load_model_):
            if self.red == 'pca':
                self.pca = pickle.load(open(out_path  +'/'+ self.robot + '_pca.sav', 'rb'))
            elif self.red == 'autoencoders':
                self.ae = autoencoder()
                self.ae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim)
                self.ae = load_model(out_path  +'/'+self.robot + '_AE',compile=False)
            # self.vae = variationalAutoencoder()
            # self.vae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim)
            elif self.red == "supervisedAutoencoders":
                self.sae = supervisedAutoencoder()
                self.sae = load_model(out_path+'/'+self.robot + '_SAE',compile=False)
            # self.svae = supervisedVariationalAutoencoder()
            # self.svae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim, 2)
            elif self.red == "supervisedClassifier":
                self.sc = load_model(out_path+'/'+self.robot + '_SC',compile=False)
        
            if self.cl == 'gmm':
                self.gmm = pickle.load(open(out_path+'/'+self.robot + '_gmm.sav', 'rb'))
            elif self.cl == 'kmeans':
                self.kmeans = pickle.load(open(out_path+'/'+self.robot + '_kmeans.sav', 'rb'))
        else:
            if self.red == 'pca':
                self.pca = PCA(n_components=self.latent_dim)
            elif self.red == 'autoencoders':
                self.ae = autoencoder()
                self.ae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim)
            # self.vae = variationalAutoencoder()
            # self.vae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim)
            elif self.red == "supervisedAutoencoders":
                self.sae = supervisedAutoencoder()
                self.sae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim, 2)
            # self.svae = supervisedVariationalAutoencoder()
            # self.svae.setDimReduction(self.input_dim, self.latent_dim, self.intermidiate_dim, 2)
            elif self.red == "supervisedClassifier":
                self.sc = supervisedClassifier()
                self.sc.setDimensions(self.input_dim, self.latent_dim, self.intermidiate_dim)
            if self.cl == 'gmm':
                self.gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter=200, tol=1.0e-3, init_params = 'kmeans', n_init=50,warm_start=False,verbose=1)
            elif self.cl == 'kmeans':
                self.kmeans = KMeans(init='k-means++',n_clusters=3, n_init=500,tol=1.0e-3)


    def setMethods(self,red, cl):
        self.red = red
        self.cl = cl

    def setFrames(self,lfoot_frame_,rfoot_frame_):
        self.lfoot_frame = lfoot_frame_
        self.rfoot_frame = rfoot_frame_

    def fit(self,data_train,data_validation, save_model_ = False, data_labels = None, data_validation_labels = None):
        print("Data Size ",data_train.size)
        self.data_train = data_train
        if self.red == 'pca':
            print("Dimensionality reduction with PCA")
            self.reducePCA(data_train, save_model_)
        elif self.red == 'autoencoders':
            print("Dimensionality reduction with autoencoders")
            self.reduceAE(data_train, data_validation, save_model_)
        # elif red == "variationalAutoencoders":
        #     print("Dimensionality reduction with variational autoencoders")
        #     self.reduceVAE(data_train, data_validation, save_model_)
        elif self.red == "supervisedAutoencoders":
            print("Dimensionality reduction with supervised autoencoders")     
            self.reduceSAE(data_train,data_labels,data_validation,data_validation_labels, save_model_)
        # elif red == "supervisedVariationalAutoencoders":
        #     print("Dimensionality reduction with supervised variational autoencoders")
        #     self.reduceSVAE(data_train,data_labels,data_validation,data_validation_labels, save_model_)
        elif self.red == "supervisedClassifier":
            print("Classification with Labels")
            self.reduceSC(data_train,data_labels,data_validation,data_validation_labels, save_model_)
        else:
            self.reduced_data_train = data_train
            print("Choose a valid dimensionality reduction method")

        
        if self.cl == 'gmm':
            print("Clustering with Gaussian Mixture Models")
            self.clusterGMM(save_model_)
        elif self.cl == 'kmeans':
            print("Clustering with Kmeans")
            self.clusterKMeans(save_model_)
        else:
            print("Choose a valid clustering method")

        self.firstrun = False



    def predict(self, data_):
        leg_probabilities = None
        support_leg = None
        if(self.red == 'pca'):
            reduced_data = self.pca.transform(data_.reshape(1,-1))
        elif(self.red == 'autoencoders'):
            reduced_data = self.ae.predict(data_.reshape(1,-1))
        # elif(self.red == 'variationalAutoencoders'):
        #     reduced_data = self.vae.encoder.predict(data_.reshape(1,-1))[0]
        elif(self.red == 'supervisedAutoencoders'):
            reduced_data = self.sae.predict(data_.reshape(1,-1))[1]
            leg_probabilities = self.sae.predict(data_.reshape(1,-1))[2]
            leg_probabilities_0 =  exp(leg_probabilities[0,0])/(exp(leg_probabilities[0,0])+exp(leg_probabilities[0,1]))
            leg_probabilities_1 =  exp(leg_probabilities[0,1])/(exp(leg_probabilities[0,0])+exp(leg_probabilities[0,1]))
            leg_probabilities[0,0] = leg_probabilities_0
            leg_probabilities[0,1] = leg_probabilities_1
            if(leg_probabilities[0,0] > leg_probabilities[0,1]):
                support_leg = self.lfoot_frame
            else:
                support_leg = self.rfoot_frame
        # elif(self.red == 'supervisedVariationalAutoencoders'):
        #     reduced_data = self.svae.encoder.predict(data_.reshape(1,-1))[0]
        elif(self.red == "supervisedClassifier"):
            reduced_data = self.sc.model.predict(data_.reshape(1,-1))[0]
        else:
            print('Unrecognired Training Method')
            reduced_data = data_

        if(self.cl == 'gmm'):
            gait_phase = self.gmm.predict(reduced_data)
            gait_phase_proba = self.gmm.predict_proba(reduced_data)
        elif(self.cl == 'kmeans'):
            gait_phase = self.kmeans.predict(reduced_data)
            gait_phase_proba = np.array([0,0,0])
        else:
            print('Unrecognired Clustering Method')

        return gait_phase, gait_phase_proba, reduced_data, leg_probabilities, support_leg


    def predict_dataset(self, data_):
        leg_probabilities = None
        if(self.red == 'pca'):
            reduced_data = self.pca.transform(data_)
        elif(self.red == 'autoencoders'):
            reduced_data = self.ae.predict(data_)
        # elif(self.red == 'variationalAutoencoders'):
        #     reduced_data = self.vae.encoder.predict(data_.reshape(1,-1))[0]
        elif(self.red == 'supervisedAutoencoders'):
            reduced_data = self.sae.predict(data_)[1]
            leg_probabilities = self.sae.predict(data_)[2]
        # elif(self.red == 'supervisedVariationalAutoencoders'):
        #     reduced_data = self.svae.encoder.predict(data_.reshape(1,-1))[0]
        elif(self.red == "supervisedClassifier"):
            reduced_data = self.sc.model.predict(data_)[0]
        else:
            print('Unrecognired Training Method')
            reduced_data = data_

        if(self.cl == 'gmm'):
            predicted_labels = self.gmm.predict(reduced_data)
        elif(self.cl == 'kmeans'):
            predicted_labels = self.kmeans.predict(reduced_data)
        else:
            print('Unrecognired Clustering Method')       

        return predicted_labels, reduced_data, leg_probabilities

    def reducePCA(self,data_train, save_model_):
        self.pca.fit(data_train)
        if(save_model_):
            with open(self.robot + '_pca.sav', 'wb') as file:
                pickle.dump(self.pca, file)       
        self.reduced_data_train = self.pca.transform(data_train)
        print("Explained variance ratio")
        print(self.pca.explained_variance_ratio_)
        print("Reprojection Error")
        print(mean_squared_error(data_train, self.pca.inverse_transform(self.reduced_data_train)))

    def reduceAE(self,data_train,data_validation, save_model_):
        self.ae.fit(data_train,data_validation,1, 2)
        self.reduced_data_train =  self.ae.encoder.predict(data_train)
        if(save_model_):
            self.ae.encoder.save(self.robot + '_AE')

    def reduceSAE(self,data_train,data_labels,data_validation,data_validation_labels, save_model_):
        self.sae.fit(data_train,data_labels,data_validation, data_validation_labels, 150, 2)
        self.reduced_data_train =  self.sae.model.predict(data_train)[1]
        self.leg_probabilities = self.sae.model.predict(data_train)[2]
        if(save_model_):
            self.sae.model.save(self.robot + '_SAE')

    # def reduceSVAE(self,data_train,data_labels,data_validation,data_validation_labels, save_model_):
    #     self.svae.fit(data_train,data_labels,data_validation, data_validation_labels, 500, 2)
    #     self.reduced_data_train =  self.svae.encoder.predict(data_train)[0]
    #     if(save_model_):
    #         self.svae.model.save(self.robot + '_SVAE.h5')

    # def reduceVAE(self,data_train,data_validation, save_model_):
    #     self.vae.fit(data_train,data_validation,50,2)
    #     self.reduced_data_train =  self.vae.encoder.predict(data_train)[0]
    #     if(save_model_):
    #         self.vae.model.save(self.robot + '_VAE.h5')

    def reduceSC(self,data_train,data_labels,data_validation,data_validation_labels, save_model_):
        self.sc.fit(data_train,data_labels,data_validation, data_validation_labels, 50, 2)
        self.reduced_data_train =  self.sc.model.predict(data_train)
        if(save_model_):
            self.sae.model.save(self.robot + '_SC')

    def clusterGMM(self, save_model_):
        self.gmm.fit(self.reduced_data_train)
        if(save_model_):
            with open(self.robot + '_gmm.sav', 'wb') as file:
                pickle.dump(self.gmm, file)
        self.predicted_labels_train = self.gmm.predict(self.reduced_data_train)


    def clusterKMeans(self, save_model_):
        self.kmeans.fit(self.reduced_data_train)
        if(save_model_):
            with open(self.robot + '_kmeans.sav', 'wb') as file:
                pickle.dump(self.kmeans, file)
        self.predicted_labels_train = self.kmeans.predict(self.reduced_data_train)

    # def getSupportLeg(self):
    #      if(self.firstrun == False):
    #         if(gait_phase == 0):
    #             self.support_leg = self.lfoot_frame
    #         elif(gait_phase == 1):
    #             self.support_leg = self.rfoot_frame
    #     else:
    #         if(data_[2]>0):
    #             self.support_leg = self.lfoot_frame
    #         else:
    #             self.support_leg = self.rfoot_frame
            
    #         self.firstrun = False
    #     return self.support_leg

    def getLLegProb(self):
        return self.pl

    def getRLegProb(self):
        return self.pr