#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 * GEM2 - Gait-phase Estimation Module 2
 *
 * Copyright 2020-2021 Stylianos Piperakis and Stavros Timotheatos, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code self.must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form self.must reproduce the above copyright
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


import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from math import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm 
from sklearn.ensemble import IsolationForest

my_colors = [(0.5,0,0.5),(0,0.5,0.5),(0.8,0.36,0.36)]
cmap_name = 'my_list'
my_cmap = LinearSegmentedColormap.from_list(
    cmap_name, my_colors, N=10000)
color_iter = itertools.cycle(my_colors)

params = {
    'axes.labelsize': 15,
    #  'text.fontsize': 15,
    'font.size' : 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'text.usetex': True,
    'figure.figsize': [7, 4] # instead of 4.5, 4.5
}
plt.rcParams.update(params)

class GEM2_tools():
    def __init__(self, validation = False, gt_comparison=False, gem2 = True):
        self.gt_comparison = gt_comparison
        self.validation = validation
        self.gem2 = gem2
        self.contamination_ = 0.25
    def input_data(self, training_path, validation_path):
        
        if(self.gt_comparison):
            phase = np.loadtxt(training_path+'/gt.txt')
            dlen = np.size(phase)
            '''
            else:
                gt_lfZ  = np.loadtxt(training_path+'/gt_lfZ.txt')
                gt_rfZ  = np.loadtxt(training_path+'/gt_rfZ.txt')
                gt_lfX  = np.loadtxt(training_path+'/gt_lfX.txt')
                gt_rfX  = np.loadtxt(training_path+'/gt_rfX.txt')
                gt_lfY  = np.loadtxt(training_path+'/gt_lfY.txt')
                gt_rfY  = np.loadtxt(training_path+'/gt_rfY.txt')
                mu  = np.loadtxt(training_path+'/mu.txt')
                dlen = min(dlen,min(np.size(gt_rfZ),np.size(gt_lfZ)))
                self.mu = mu
                lcon = np.sqrt(gt_lfX[i] * gt_lfX[i] + gt_lfY[i] * gt_lfY[i])
                rcon = np.sqrt(gt_rfX[i] * gt_rfX[i] + gt_rfY[i] * gt_rfY[i])
                if( ((self.mu[i]*gt_lfZ[i])>lcon) and ((self.mu[i] * gt_rfZ[i])>rcon)):
                    phase[i] = 0
                elif( (self.mu[i]*gt_lfZ[i])>lcon ):
                    phase[i] = 1
                elif( (self.mu[i]*gt_rfZ[i])>rcon ):
                    phase[i] = 2
                else:
                    phase[i] = -1
            '''



        rfX = np.loadtxt(training_path+'/rfX.txt')
        rfY = np.loadtxt(training_path+'/rfY.txt')
        rfZ = np.loadtxt(training_path+'/rfZ.txt')
        rtX = np.loadtxt(training_path+'/rtX.txt')
        rtY = np.loadtxt(training_path+'/rtY.txt')
        rtZ = np.loadtxt(training_path+'/rtZ.txt')
        lfX = np.loadtxt(training_path+'/lfX.txt')
        lfY = np.loadtxt(training_path+'/lfY.txt')
        lfZ = np.loadtxt(training_path+'/lfZ.txt')
        ltX = np.loadtxt(training_path+'/ltX.txt')
        ltY = np.loadtxt(training_path+'/ltY.txt')
        ltZ = np.loadtxt(training_path+'/ltZ.txt')
        dlen = min(np.size(lfZ),np.size(rfZ))
        gX = np.loadtxt(training_path+'/gX.txt')
        gY = np.loadtxt(training_path+'/gY.txt')
        gZ = np.loadtxt(training_path+'/gZ.txt')
        accX = np.loadtxt(training_path+'/accX.txt')
        accY = np.loadtxt(training_path+'/accY.txt')
        accZ = np.loadtxt(training_path+'/accZ.txt')
        dlen = min(dlen,np.size(accZ))
        dcX = np.loadtxt(training_path+'/comvX.txt')
        dcY = np.loadtxt(training_path+'/comvY.txt')
        dcZ = np.loadtxt(training_path+'/comvZ.txt')
        dlen = min(dlen,np.size(dcZ))      

        if(self.gem2):
            lvX = np.loadtxt(training_path+'/lvX.txt')
            lvY = np.loadtxt(training_path+'/lvY.txt')
            lvZ = np.loadtxt(training_path+'/lvZ.txt')
            dlen = min(dlen,np.size(lvZ))
            rvX = np.loadtxt(training_path+'/rvX.txt')
            rvY = np.loadtxt(training_path+'/rvY.txt')
            rvZ = np.loadtxt(training_path+'/rvZ.txt')
            dlen = min(dlen,np.size(rvZ))
            lwX = np.loadtxt(training_path+'/lwX.txt')
            lwY = np.loadtxt(training_path+'/lwY.txt')
            lwZ = np.loadtxt(training_path+'/lwZ.txt')
            rwX = np.loadtxt(training_path+'/rwX.txt')
            rwY = np.loadtxt(training_path+'/rwY.txt')
            rwZ = np.loadtxt(training_path+'/rwZ.txt')
            #laccX = np.loadtxt(training_path+'/laccX.txt')
            #laccY = np.loadtxt(training_path+'/laccY.txt')
            #laccZ = np.loadtxt(training_path+'/laccZ.txt')
            #dlen = min(dlen,np.size(laccZ))
            #raccX = np.loadtxt(training_path+'/raccX.txt')
            #raccY = np.loadtxt(training_path+'/raccY.txt')
            #raccZ = np.loadtxt(training_path+'/raccZ.txt')
            #dlen = min(dlen,np.size(raccZ))
            baccX_LL = np.loadtxt(training_path+'/baccX_LL.txt')
            baccY_LL = np.loadtxt(training_path+'/baccY_LL.txt')
            baccZ_LL = np.loadtxt(training_path+'/baccZ_LL.txt')
            baccX_RL = np.loadtxt(training_path+'/baccX_RL.txt')
            baccY_RL = np.loadtxt(training_path+'/baccY_RL.txt')
            baccZ_RL = np.loadtxt(training_path+'/baccZ_RL.txt')
            baccX = np.loadtxt(training_path+'/baccX.txt')
            baccY = np.loadtxt(training_path+'/baccY.txt')
            baccZ = np.loadtxt(training_path+'/baccZ.txt')

            bgX_LL = np.loadtxt(training_path+'/bgX_LL.txt')
            bgY_LL = np.loadtxt(training_path+'/bgY_LL.txt')
            bgZ_LL = np.loadtxt(training_path+'/bgZ_LL.txt')
            bgX_RL = np.loadtxt(training_path+'/bgX_RL.txt')
            bgY_RL = np.loadtxt(training_path+'/bgY_RL.txt')
            bgZ_RL = np.loadtxt(training_path+'/bgZ_RL.txt')
            bgX = np.loadtxt(training_path+'/bgX.txt')
            bgY = np.loadtxt(training_path+'/bgY.txt')
            bgZ = np.loadtxt(training_path+'/bgZ.txt')
            dlen = min(dlen,min(np.size(bgZ_LL),np.size(baccZ_RL)))
       


        if(self.validation):
            rfX_val = np.loadtxt(validation_path+'/rfX.txt')
            rfY_val = np.loadtxt(validation_path+'/rfY.txt')
            rfZ_val = np.loadtxt(validation_path+'/rfZ.txt')
            rtX_val = np.loadtxt(validation_path+'/rtX.txt')
            rtY_val = np.loadtxt(validation_path+'/rtY.txt')
            rtZ_val = np.loadtxt(validation_path+'/rtZ.txt')
            lfX_val = np.loadtxt(validation_path+'/lfX.txt')
            lfY_val = np.loadtxt(validation_path+'/lfY.txt')
            lfZ_val = np.loadtxt(validation_path+'/lfZ.txt')
            ltX_val = np.loadtxt(validation_path+'/ltX.txt')
            ltY_val = np.loadtxt(validation_path+'/ltY.txt')
            ltZ_val = np.loadtxt(validation_path+'/ltZ.txt')
            dlen_val = min(np.size(lfZ_val),np.size(rfZ_val))
            gX_val = np.loadtxt(validation_path+'/gX.txt')
            gY_val = np.loadtxt(validation_path+'/gY.txt')
            gZ_val = np.loadtxt(validation_path+'/gZ.txt')
            accX_val = np.loadtxt(validation_path+'/accX.txt')
            accY_val = np.loadtxt(validation_path+'/accY.txt')
            accZ_val = np.loadtxt(validation_path+'/accZ.txt')
            dlen_val = min(dlen_val,np.size(accZ_val))
            dcX_val = np.loadtxt(validation_path+'/comvX.txt')
            dcY_val = np.loadtxt(validation_path+'/comvY.txt')
            dcZ_val = np.loadtxt(validation_path+'/comvZ.txt')
            dlen_val = min(dlen_val,np.size(dcZ_val))            

            if(self.gem2):
                lvX_val = np.loadtxt(validation_path+'/lvX.txt')
                lvY_val = np.loadtxt(validation_path+'/lvY.txt')
                lvZ_val = np.loadtxt(validation_path+'/lvZ.txt')
                dlen_val = min(dlen_val,np.size(lvZ_val))
                rvX_val = np.loadtxt(validation_path+'/rvX.txt')
                rvY_val = np.loadtxt(validation_path+'/rvY.txt')
                rvZ_val = np.loadtxt(validation_path+'/rvZ.txt')
                dlen_val = min(dlen_val,np.size(rvZ_val))
                lwX_val = np.loadtxt(validation_path+'/lwX.txt')
                lwY_val = np.loadtxt(validation_path+'/lwY.txt')
                lwZ_val = np.loadtxt(validation_path+'/lwZ.txt')
                rwX_val = np.loadtxt(validation_path+'/rwX.txt')
                rwY_val = np.loadtxt(validation_path+'/rwY.txt')
                rwZ_val = np.loadtxt(validation_path+'/rwZ.txt')
                #laccX_val = np.loadtxt(validation_path+'/laccX.txt')
                #laccY_val = np.loadtxt(validation_path+'/laccY.txt')
                #laccZ_val = np.loadtxt(validation_path+'/laccZ.txt')
                #dlen_val = min(dlen_val,np.size(laccZ_val))
                #raccX_val = np.loadtxt(validation_path+'/raccX.txt')
                #raccY_val = np.loadtxt(validation_path+'/raccY.txt')
                #raccZ_val = np.loadtxt(validation_path+'/raccZ.txt')
                #dlen_val = min(dlen_val,np.size(raccZ_val))
                baccX_LL_val = np.loadtxt(validation_path+'/baccX_LL.txt')
                baccY_LL_val = np.loadtxt(validation_path+'/baccY_LL.txt')
                baccZ_LL_val = np.loadtxt(validation_path+'/baccZ_LL.txt')
                baccX_RL_val = np.loadtxt(validation_path+'/baccX_RL.txt')
                baccY_RL_val = np.loadtxt(validation_path+'/baccY_RL.txt')
                baccZ_RL_val = np.loadtxt(validation_path+'/baccZ_RL.txt')
                baccX_val = np.loadtxt(validation_path+'/baccX.txt')
                baccY_val = np.loadtxt(validation_path+'/baccY.txt')
                baccZ_val = np.loadtxt(validation_path+'/baccZ.txt')
                bgX_LL_val = np.loadtxt(training_path+'/bgX_LL.txt')
                bgY_LL_val = np.loadtxt(training_path+'/bgY_LL.txt')
                bgZ_LL_val = np.loadtxt(training_path+'/bgZ_LL.txt')
                bgX_RL_val = np.loadtxt(training_path+'/bgX_RL.txt')
                bgY_RL_val = np.loadtxt(training_path+'/bgY_RL.txt')
                bgZ_RL_val = np.loadtxt(training_path+'/bgZ_RL.txt')
                bgX_val = np.loadtxt(training_path+'/bgX.txt')
                bgY_val = np.loadtxt(training_path+'/bgY.txt')
                bgZ_val = np.loadtxt(training_path+'/bgZ.txt')
                dlen_val = min(dlen_val,min(np.size(bgZ_LL_val),np.size(baccZ_RL_val)))
            


        self.data_label = np.array([])
        self.data_train = np.array([])
        self.data_val = np.array([])
        self.data_val_label = np.array([])

        X_train = np.array([])
        Y_train = np.array([])
        X_val = np.array([])
        Y_val = np.array([])

        #Leg Forces and Torques
        if(not self.gem2):
            X_train = lfX[0:dlen] - rfX[0:dlen]
            X_train = np.column_stack([X_train, lfY[0:dlen] - rfY[0:dlen]])
            X_train = np.column_stack([X_train, lfZ[0:dlen] - rfZ[0:dlen]])
            X_train = np.column_stack([X_train, ltX[0:dlen] - rtX[0:dlen]])
            X_train = np.column_stack([X_train, ltY[0:dlen] - rtY[0:dlen]])
            X_train = np.column_stack([X_train, ltZ[0:dlen] - rtZ[0:dlen]])
            #CoM Velocity
            X_train = np.column_stack([X_train, dcX[0:dlen]])
            X_train = np.column_stack([X_train, dcY[0:dlen]])
            X_train = np.column_stack([X_train, dcZ[0:dlen]])
            #Base Linear Acceleration and Base Angular Velocity
            X_train = np.column_stack([X_train, accX[0:dlen]])
            X_train = np.column_stack([X_train, accY[0:dlen]])
            X_train = np.column_stack([X_train, accZ[0:dlen]])
            X_train = np.column_stack([X_train, gX[0:dlen]])
            X_train = np.column_stack([X_train, gY[0:dlen]])
            X_train = np.column_stack([X_train, gZ[0:dlen]])



            iso = IsolationForest(contamination=self.contamination_)    
            yhat = iso.fit_predict(X_train)          
            self.outlier_mask = yhat != -1
            self.data_train = X_train[self.outlier_mask,:]
        else:
            #X_train = lfX[0:dlen] - rfX[0:dlen]
            #X_train = np.column_stack([X_train, lfY[0:dlen] - rfY[0:dlen]])
            #X_train = np.column_stack([X_train, lfZ[0:dlen] - rfZ[0:dlen]])
            #X_train = np.column_stack([X_train, ltX[0:dlen] - rtX[0:dlen]])
            #X_train = np.column_stack([X_train, ltY[0:dlen] - rtY[0:dlen]])
            #X_train = np.column_stack([X_train, ltZ[0:dlen] - rtZ[0:dlen]])
            X_train = np.abs(lfX[0:dlen])
            X_train = np.column_stack([X_train, np.abs(lfY[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(lfZ[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(ltX[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(ltY[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(ltZ[0:dlen])])


            #CoM Velocity
            #X_train = np.column_stack([X_train, dcX[0:dlen]])
            #X_train = np.column_stack([X_train, dcY[0:dlen]])
            #X_train = np.column_stack([X_train, dcZ[0:dlen]])
            #Base Linear Acceleration and Base Angular Velocity
            #X_train = np.column_stack([X_train, accX[0:dlen]])
            #X_train = np.column_stack([X_train, accY[0:dlen]])
            #X_train = np.column_stack([X_train, accZ[0:dlen]])
            #X_train = np.column_stack([X_train, gX[0:dlen]])
            #X_train = np.column_stack([X_train, gY[0:dlen]])
            #X_train = np.column_stack([X_train, gZ[0:dlen]])

            X_train = np.column_stack([X_train, np.abs(rfX[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(rfY[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(rfZ[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(rtX[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(rtY[0:dlen])])
            X_train = np.column_stack([X_train, np.abs(rtZ[0:dlen])])



            #Base/Legs Acceleration as labels
            Y_train = np.abs(bgX_LL[0:dlen])
            Y_train = np.column_stack([Y_train,np.abs(bgY_LL[0:dlen])])
            Y_train = np.column_stack([Y_train,np.abs(bgZ_LL[0:dlen])])
            
            Y_train = np.column_stack([Y_train,np.abs(bgX_RL[0:dlen])])
            Y_train = np.column_stack([Y_train,np.abs(bgY_RL[0:dlen])])
            Y_train = np.column_stack([Y_train,np.abs(bgZ_RL[0:dlen])])


            Y_train = np.column_stack([Y_train,np.abs(baccX_LL[0:dlen])])
            Y_train = np.column_stack([Y_train,np.abs(baccY_LL[0:dlen])])
            Y_train = np.column_stack([Y_train,np.abs(baccZ_LL[0:dlen])])

            Y_train = np.column_stack([Y_train,np.abs(baccX_RL[0:dlen])])
            Y_train = np.column_stack([Y_train,np.abs(baccY_RL[0:dlen])])
            Y_train = np.column_stack([Y_train,np.abs(baccZ_RL[0:dlen])])
            

            iso = IsolationForest(contamination=self.contamination_)    
            yhat = iso.fit_predict(np.column_stack([X_train,Y_train]))          
            self.outlier_mask = yhat != -1
            self.data_train = X_train[self.outlier_mask,:]
            data_label = Y_train[self.outlier_mask,:]



            self.data_train_min = np.zeros((self.data_train.shape[1]))
            self.data_train_max = np.zeros((self.data_train.shape[1]))
            self.data_train_mean = np.zeros((self.data_train.shape[1]))
            self.data_train_std = np.zeros((self.data_train.shape[1]))

            #Data Statistics
            for i in range(self.data_train.shape[1]):
                self.data_train_min[i] = np.min(self.data_train[:, i])
                self.data_train_max[i] = np.max(self.data_train[:, i])
                self.data_train_mean[i] = np.mean(self.data_train[:, i])
                self.data_train_std[i] = np.std(self.data_train[:, i])
                self.data_train[:, i] = self.normalize_data(self.data_train[:, i],self.data_train_max[i], self.data_train_min[i],0,1)   
                #self.data_train[:, i] = self.standarize_data(self.data_train[:, i],self.data_train_mean[i], self.data_train_std[i])   
                #self.data_train[:, i] = self.normalizeMean_data(self.data_train[:, i],self.data_train_max[i], self.data_train_min[i],self.data_train_mean[i])   




            data_label_min = np.zeros((data_label.shape[1]))
            data_label_max = np.zeros((data_label.shape[1]))
            #Label Statistics
            for i in range(data_label.shape[1]):
                data_label_min[i] = np.min(data_label[:, i])
                data_label_max[i] = np.max(data_label[:, i])
                data_label[:, i] = 1-self.normalize_data(data_label[:, i],data_label_max[i], data_label_min[i], 0 , 1) 



            #Left Leg
            tmpLw = np.multiply(data_label[:, 0],data_label[:, 1])
            tmpLw = np.multiply(tmpLw,data_label[:, 2])

            tmpLa = np.multiply(data_label[:, 6],data_label[:, 7])
            tmpLa = np.multiply(tmpLa,data_label[:, 8])

            tmpLa = np.multiply(tmpLa,self.data_train[:,2])
            tmpL = np.multiply(tmpLw,tmpLa)
            tmpL_ = norm.cdf(tmpL,np.mean(tmpL),np.std(tmpL))
            tmpL_ = self.normalize_data(tmpL_,np.max(tmpL_),np.min(tmpL_),0,1)
            self.data_label = tmpL_


            #Right Leg
            tmpRw = np.multiply(data_label[:, 3],data_label[:, 4])
            tmpRw = np.multiply(tmpRw,data_label[:, 5])

            tmpRa = np.multiply(data_label[:, 9],data_label[:, 10])
            tmpRa = np.multiply(tmpRa,data_label[:, 11])

            tmpRa = np.multiply(tmpRa,self.data_train[:,8])
            tmpR = np.multiply(tmpRw,tmpRa)
            tmpR_ = norm.cdf(tmpR,np.mean(tmpR),np.std(tmpR))
            tmpR_ = self.normalize_data(tmpR_,np.max(tmpR_),np.min(tmpR_),0,1)
            self.data_label = np.column_stack([self.data_label,tmpR_])

            #plt.plot(self.data_label[:,0])
            #plt.plot(self.data_label[:,1])
            #plt.show()

        if(self.validation):            
            if(not self.gem2):
                #Leg Forces and Torques
                X_val = lfX_val[0:dlen_val] - rfX_val[0:dlen_val]
                X_val = np.column_stack([X_val, lfY_val[0:dlen_val] - rfY_val[0:dlen_val]])
                X_val = np.column_stack([X_val, lfZ_val[0:dlen_val] - rfZ_val[0:dlen_val]])
                X_val = np.column_stack([X_val, ltX_val[0:dlen_val] - rtX_val[0:dlen_val]])
                X_val = np.column_stack([X_val, ltY_val[0:dlen_val] - rtY_val[0:dlen_val]])
                X_val = np.column_stack([X_val, ltZ_val[0:dlen_val] - rtZ_val[0:dlen_val]])
                #CoM Velocity
                #X_val = np.column_stack([X_val, dcX_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, dcY_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, dcZ_val[0:dlen_val]])
                #Base Linear Acceleration and Base Angular Velocity                
                X_val = np.column_stack([X_val, accX_val[0:dlen_val]])
                X_val = np.column_stack([X_val, accY_val[0:dlen_val]])
                X_val = np.column_stack([X_val, accZ_val[0:dlen_val]])
                X_val = np.column_stack([X_val, gX_val[0:dlen_val]])
                X_val = np.column_stack([X_val, gY_val[0:dlen_val]])
                X_val = np.column_stack([X_val, gZ_val[0:dlen_val]])
                iso = IsolationForest(contamination=self.contamination_)    
                yhat = iso.fit_predict(X_val) 
                self.outlier_mask_val = yhat != -1
                self.data_val = X_val[self.outlier_mask_val,:]


            else:
                # X_val = lfX_val[0:dlen_val] - rfX_val[0:dlen_val]
                # X_val = np.column_stack([X_val, lfY_val[0:dlen_val] - rfY_val[0:dlen_val]])
                # X_val = np.column_stack([X_val, lfZ_val[0:dlen_val] - rfZ_val[0:dlen_val]])
                # X_val = np.column_stack([X_val, ltX_val[0:dlen_val] - rtX_val[0:dlen_val]])
                # X_val = np.column_stack([X_val, ltY_val[0:dlen_val] - rtY_val[0:dlen_val]])
                # X_val = np.column_stack([X_val, ltZ_val[0:dlen_val] - rtZ_val[0:dlen_val]])
                X_val = np.abs(lfX_val[0:dlen_val])
                X_val = np.column_stack([X_val, np.abs(lfY_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(lfZ_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(ltX_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(ltY_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(ltZ_val[0:dlen_val])])


                #CoM Velocity
                #X_val = np.column_stack([X_val, dcX_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, dcY_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, dcZ_val[0:dlen_val]])
                #Base Linear Acceleration and Base Angular Velocity                
                #X_val = np.column_stack([X_val, accX_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, accY_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, accZ_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, gX_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, gY_val[0:dlen_val]])
                #X_val = np.column_stack([X_val, gZ_val[0:dlen_val]])


                X_val = np.column_stack([X_val, np.abs(rfX_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(rfY_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(rfZ_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(rtX_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(rtY_val[0:dlen_val])])
                X_val = np.column_stack([X_val, np.abs(rtZ_val[0:dlen_val])])




                #Base/Legs Acceleration as labels
                Y_val = np.abs(bgX_LL_val[0:dlen_val])
                Y_val = np.column_stack([Y_val, np.abs(bgY_LL_val[0:dlen_val])])
                Y_val = np.column_stack([Y_val, np.abs(bgZ_LL_val[0:dlen_val])])
                
                Y_val = np.column_stack([Y_val, np.abs(bgX_RL_val[0:dlen_val])])
                Y_val = np.column_stack([Y_val, np.abs(bgY_RL_val[0:dlen_val])])
                Y_val = np.column_stack([Y_val, np.abs(bgZ_RL_val[0:dlen_val])])


                Y_val = np.column_stack([Y_val, np.abs(baccX_LL_val[0:dlen_val])])
                Y_val = np.column_stack([Y_val, np.abs(baccY_LL_val[0:dlen_val])])
                Y_val = np.column_stack([Y_val, np.abs(baccZ_LL_val[0:dlen_val])])

                Y_val = np.column_stack([Y_val, np.abs(baccX_RL_val[0:dlen_val])])
                Y_val = np.column_stack([Y_val, np.abs(baccY_RL_val[0:dlen_val])])
                Y_val = np.column_stack([Y_val, np.abs(baccZ_RL_val[0:dlen_val])])

                iso = IsolationForest(contamination=self.contamination_)    
                yhat = iso.fit_predict(np.column_stack([X_val,Y_val]))          
                self.outlier_mask_val = yhat != -1
                self.data_val = X_val[self.outlier_mask_val,:]
                data_val_label = Y_val[self.outlier_mask_val,:]


                #Normalize Validation Data
                self.data_val_min = np.zeros((self.data_val.shape[1]))
                self.data_val_max = np.zeros((self.data_val.shape[1]))
                self.data_val_mean = np.zeros((self.data_val.shape[1]))
                self.data_val_std = np.zeros((self.data_val.shape[1]))
            
                #Data Statistics
                for i in range(self.data_val.shape[1]):
                    self.data_val_min[i] = np.min(self.data_val[:, i])
                    self.data_val_max[i] = np.max(self.data_val[:, i])
                    self.data_val_mean[i] = np.mean(self.data_val[:, i])
                    self.data_val_std[i] = np.std(self.data_val[:, i])
                    self.data_val[:, i] = self.normalize_data(self.data_val[:, i],self.data_val_max[i], self.data_val_min[i],0,1)


                for i in range(data_val_label.shape[1]):
                    data_val_label[:, i] = 1-self.normalize_data(data_val_label[:, i],np.max(data_val_label[:, i]), np.min(data_val_label[:, i]),0,1)   


 
                tmpLw = np.multiply(data_val_label[:, 0],data_val_label[:, 1])
                tmpLw = np.multiply(tmpLw,data_val_label[:, 2])

                tmpLa = np.multiply(data_val_label[:, 6],data_val_label[:, 7])
                tmpLa = np.multiply(tmpLa,data_val_label[:, 8])

                tmpLa = np.multiply(tmpLa,self.data_val[:,2])
                tmpL = np.multiply(tmpLw,tmpLa)
            
                tmpL_ = norm.cdf(tmpL,np.mean(tmpL),np.std(tmpL))
                tmpL_ = self.normalize_data(tmpL_,np.max(tmpL_),np.min(tmpL_),0,1)
                self.data_val_label = tmpL_


                #Right Leg
                tmpRw = np.multiply(data_val_label[:, 3],data_val_label[:, 4])
                tmpRw = np.multiply(tmpRw,data_val_label[:, 5])

                tmpRa = np.multiply(data_val_label[:, 9],data_val_label[:, 10])
                tmpRa = np.multiply(tmpRa,data_val_label[:, 11])

                tmpRa = np.multiply(tmpRa,self.data_val[:,8])
                tmpR = np.multiply(tmpRw,tmpRa)
                tmpR_ = norm.cdf(tmpR,np.mean(tmpR),np.std(tmpR))
                tmpR_ = self.normalize_data(tmpR_,np.max(tmpR_),np.min(tmpR_),0,1)
                self.data_val_label = np.column_stack([self.data_val_label,tmpR_])

                #plt.plot(self.data_val_label[:,0])
                #plt.plot(self.data_val_label[:,1])
                #plt.show()





        if (self.gt_comparison):
            self.phase = phase[self.outlier_mask]
            '''
            else:
                phase2=np.append([phase],[np.zeros_like(np.arange(cX.shape[0]-phase.shape[0]))])
                self.cX = cX[~(phase2==-1)]
                self.cY = cY[~(phase2==-1)]
                self.cZ = cZ[~(phase2==-1)]
                phase3=np.append([phase],[np.zeros_like(np.arange(accX.shape[0]-phase.shape[0]))])
                self.accX = accX[~(phase3==-1)]
                self.accY = accY[~(phase3==-1)]
                self.accZ = accZ[~(phase3==-1)]
                phase4=np.append([phase],[np.zeros_like(np.arange(gX.shape[0]-phase.shape[0]))])
                self.gX = gX[~(phase4==-1)]
                self.gY = gY[~(phase4==-1)]
                phase5=np.append([phase],[np.zeros_like(np.arange(lfZ.shape[0]-phase.shape[0]))])
                self.lfZ = lfZ[~(phase5==-1)]
                self.lfX = lfX[~(phase5==-1)]
                self.lfY = lfY[~(phase5==-1)]
                phase6=np.append([phase],[np.zeros_like(np.arange(rfZ.shape[0]-phase.shape[0]))])
                self.rfZ = rfZ[~(phase6==-1)]
                self.rfX = rfX[~(phase6==-1)]
                self.rfY = rfY[~(phase6==-1)]
                phase7=np.append([phase],[np.zeros_like(np.arange(ltZ.shape[0]-phase.shape[0]))])
                self.ltZ = ltZ[~(phase7==-1)]
                self.ltX = ltX[~(phase7==-1)]
                self.ltY = ltY[~(phase7==-1)]
                phase8=np.append([phase],[np.zeros_like(np.arange(rtZ.shape[0]-phase.shape[0]))])
                self.rtZ = rtZ[~(phase8==-1)]
                self.rtX = rtX[~(phase8==-1)]
                self.rtY = rtY[~(phase8==-1)]
                self.data_train=self.data_train[~(phase==-1)]
                self.phase=phase[~(phase==-1)]
                self.dlen = np.size(self.phase)
            '''

        self.dlen = self.data_train.shape[0]
        

        print("Data Dim Train")
        print(np.shape(self.data_train))
        print("Data Dim Validation")
        print(np.shape(self.data_val))

    def genInput(self, data):


        output_ = np.array([])
        output_ = np.insert(output_, 0, data.lfX - data.rfX, axis = 0)
        output_ = np.insert(output_, 1, data.lfY - data.rfY, axis = 0)
        output_ = np.insert(output_, 2, data.lfZ - data.rfZ, axis = 0)
        output_ = np.insert(output_, 3, data.ltX - data.rtX, axis = 0)
        output_ = np.insert(output_, 4, data.ltY - data.rtY, axis = 0)
        output_ = np.insert(output_, 5, data.ltZ - data.rtZ, axis = 0)
        output_ = np.insert(output_, 6, data.dcX, axis = 0)
        output_ = np.insert(output_, 7, data.dcY, axis = 0)
        output_ = np.insert(output_, 8, data.dcZ, axis = 0)


        if(self.gem2):
            output_ = np.insert(output_, 9, data.lvX - data.rvX, axis = 0)
            output_ = np.insert(output_, 10, data.lvY - data.rvY, axis = 0)
            output_ = np.insert(output_, 11, data.lvZ - data.rvZ, axis = 0)
            output_ = np.insert(output_, 12, data.lwX - data.rwX, axis = 0)
            output_ = np.insert(output_, 13, data.lwY - data.rwY, axis = 0)
            output_ = np.insert(output_, 14, data.lwZ - data.rwZ, axis = 0)
            
            output_ = np.insert(output_, 15, data.accX, axis = 0)
            output_ = np.insert(output_, 16, data.accY, axis = 0)
            output_ = np.insert(output_, 17, data.accZ, axis = 0)
            output_ = np.insert(output_, 18, data.gX, axis = 0)
            output_ = np.insert(output_, 19, data.gY, axis = 0)
            output_ = np.insert(output_, 20, data.gZ, axis = 0)
        else:
            output_ = np.insert(output_, 9, data.accX, axis = 0)
            output_ = np.insert(output_, 10, data.accY, axis = 0)
            output_ = np.insert(output_, 11, data.accZ, axis = 0)
            output_ = np.insert(output_, 12, data.gX, axis = 0)
            output_ = np.insert(output_, 13, data.gY, axis = 0)
            output_ = np.insert(output_, 14, data.gZ, axis = 0)

       
        for i in range(self.data_train.shape[1]):
            output_[i] = self.normalize(output_[i],self.data_train_max[i], self.data_train_min[i])   
            #output_[i] = self.normalizeMean_data(output_[i],self.data_train_max[i], self.data_train_min[i], self.data_train_mean[i])   

       
        return output_


    def normalize_data(self,din, dmax, dmin, min_range=-1, max_range = 1):    
        if(dmax-dmin != 0):
            dout = min_range  + (max_range - min_range) * (din - dmin)/(dmax - dmin)
        else:
            dout =  np.zeros((np.size(din)))

        return dout

    def standarize_data(self,din,dmean,dstd):
        if(dstd != 0):
            dout = (din - dmean)/dstd
        else:
            dout =  np.zeros((np.size(din)))

        return dout


    def normalize(self,din, dmax, dmin, min_range=-1, max_range = 1):    
        if(din>dmax):
            din=dmax
        elif(din<dmin):
            din=dmin

        if(dmax-dmin != 0):
            dout = min_range  + (max_range - min_range) * (din - dmin)/(dmax - dmin)
        else:
            dout =  0

        return dout

    def normalizeMean(self,din, dmax, dmin, dmean):    
        if(din>dmax):
            din=dmax
        elif(din<dmin):
            din=dmin

        if(dmax-dmin != 0):
            dout = (din - dmean)/(dmax-dmin)
        else:
            dout =  0

        return dout

    def normalizeMean_data(self,din, dmax, dmin, dmean):    
        if(dmax-dmin != 0):
            dout = (din - dmean)/(dmax-dmin)
        else:
            dout =  np.zeros((np.size(din)))

        return dout  

    def standarize(self,din,dmean,dstd):
        if(dstd != 0):
            dout = (din - dmean)/dstd
        else:
            dout =  0

        return dout


    def genGroundTruthStatistics(self, reduced_data):
        if(self.gt_comparison):
            #remove extra zeros elements
            d1 = np.zeros((self.dlen,2))
            d2 = np.zeros((self.dlen,2))
            d3 = np.zeros((self.dlen,2))
            
            for i in range(self.dlen):
                if (self.phase[i]==0):
                    d1[i,0] = reduced_data[i,0]
                    d1[i,1] = reduced_data[i,1]
                elif (self.phase[i]==1):
                    d2[i,0] = reduced_data[i,0]
                    d2[i,1] = reduced_data[i,1]
                elif (self.phase[i]==2):
                    d3[i,0] = reduced_data[i,0]
                    d3[i,1] = reduced_data[i,1]

            d1=d1[~(d1==0).all(1)]
            d2=d2[~(d2==0).all(1)]
            d3=d3[~(d3==0).all(1)]
            # print('----')
            # print(d1)
            # print('----')
            # print('----')
            # print(d2)
            # print('----')
            # print('----')
            # print(d3)
            # print('----')
            mean=np.zeros((3,2))
            mean[0,0]=np.mean(d1[:,0])
            mean[0,1]=np.mean(d1[:,1])
            mean[1,0]=np.mean(d2[:,0])
            mean[1,1]=np.mean(d2[:,1])
            mean[2,0]=np.mean(d3[:,0])
            mean[2,1]=np.mean(d3[:,1])

            print(mean)

            self.mean = mean
            covariance1=np.cov(d1.T)
            covariance2=np.cov(d2.T)
            covariance3=np.cov(d3.T)
            self.covariance=(covariance1, covariance2, covariance3)
        else:
            print('Input data did not have Ground-Truth Information')



    def plot_accelerations_LR(self,leg_probabilities, data_labels):
        t = np.arange(0,leg_probabilities.shape[0], 1)
        base_accX = data_labels[:,6]
        base_accY = data_labels[:,7]
        base_accZ = data_labels[:,8]
        
        #est_accX = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,9]) + np.multiply(leg_probabilities[:,1],data_labels[:,12]) ), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        #est_accY = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,10]) + np.multiply(leg_probabilities[:,1],data_labels[:,13])), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        #est_accZ = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,11]) + np.multiply(leg_probabilities[:,1],data_labels[:,14])), (leg_probabilities[:,0] + leg_probabilities[:,1]))

        est_accX_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], base_accX)
        est_accX_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], base_accX)
        est_accY_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], base_accY)
        est_accY_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], base_accY)
        est_accZ_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], base_accZ)
        est_accZ_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], base_accZ)

        plt.figure()
        plt.subplot(311)
        plt.plot(t,base_accX)
        #plt.scatter(t,est_accX,s=2.0, color=c)
        plt.plot(t,est_accX_LL,t,est_accX_RL)
        plt.ylabel('$acc_x$')
        plt.subplot(312)
        plt.plot(t,base_accY)
        #plt.scatter(t,est_accY, s=2.0,color=c)
        plt.plot(t,est_accY_LL,t,est_accY_RL)
        plt.ylabel('$acc_y$')
        plt.subplot(313)
        plt.plot(t,base_accZ)
        #plt.scatter(t,est_accZ, s=2.0,color=c)
        plt.plot(t,est_accZ_LL,t,est_accZ_RL)
        plt.ylabel('$acc_z$')
        plt.xlabel('$samples$')
        plt.show()

        base_gX = data_labels[:,9]
        base_gY = data_labels[:,10]
        base_gZ = data_labels[:,11]
        
        #est_gX = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,0]) + np.multiply(leg_probabilities[:,1],data_labels[:,3])), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        #est_gY = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,1]) + np.multiply(leg_probabilities[:,1],data_labels[:,4])), (leg_probabilities[:,0] + leg_probabilities[:,1]))
        #est_gZ = np.divide( (np.multiply(leg_probabilities[:,0], data_labels[:,2]) + np.multiply(leg_probabilities[:,1],data_labels[:,5])), (leg_probabilities[:,0] + leg_probabilities[:,1]))

        est_gX_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], base_gX)
        est_gX_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], base_gX)
        est_gY_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], base_gY)
        est_gY_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], base_gY)
        est_gZ_LL = np.ma.masked_where(leg_probabilities[:,0] < leg_probabilities[:,1], base_gZ)
        est_gZ_RL = np.ma.masked_where(leg_probabilities[:,0] >= leg_probabilities[:,1], base_gZ)

        plt.figure()
        plt.subplot(311)
        plt.plot(t,base_gX)
        #plt.scatter(t,est_gX,s=2.0, color=c)
        plt.plot(t,est_gX_LL,t,est_gX_RL)
        plt.ylabel('$g_x$')
        plt.subplot(312)
        plt.plot(t,base_gY)
        #plt.scatter(t,est_gY, s=2.0,color=c)
        plt.plot(t,est_gY_LL,t,est_gY_RL)
        plt.ylabel('$g_y$')
        plt.subplot(313)
        plt.plot(t,base_gZ)
        #plt.scatter(t,est_gZ, s=2.0,color=c)
        plt.plot(t,est_gZ_LL,t,est_gZ_RL)
        plt.ylabel('$g_z$')
        plt.xlabel('$samples$')
        plt.show()


    def plot_reconstruction(self,data_in, data_out):

        number_of_subplots=data_in.shape[1]

        for i,v in enumerate(xrange(number_of_subplots)):
            v = v+1
            ax1 = plt.subplot(number_of_subplots,1,v)
            ax1.plot(data_in[:,i])
            ax1.plot(data_out[:,i])

        plt.show()

    def plot_accelerations_LRD(self,leg_probabilities, predicted_labels,  x):
        t = np.arange(0,x.shape[0], 1)
        eps = 0.1

        #LSS_mask = np.where( np.logical_and( predicted_labels > 0-eps,  predicted_labels<0+eps))[0]
        #RSS_mask = np.where( np.logical_and( predicted_labels > 1-eps,  predicted_labels<1+eps))[0]
        #DS_mask = np.where( np.logical_and( predicted_labels > 2-eps,  predicted_labels<2+eps))[0]

     
        x_DS =  np.isclose(predicted_labels, np.zeros(np.shape(predicted_labels)), atol=eps)
        x_RSS =  np.isclose(predicted_labels, np.ones(np.shape(predicted_labels)), atol=eps)
        x_LSS =   np.isclose(predicted_labels, 2.0*np.ones(np.shape(predicted_labels)), atol=eps)

        y_LSS = np.copy(x[:,0])
        y_LSS[~x_LSS] = np.nan


        y_RSS = np.copy(x[:,0])
        y_RSS[~x_RSS] = np.nan

        
        y_DS = np.copy(x[:,0])
        y_DS[~x_DS] = np.nan

        plt.figure()
        plt.subplot(311)
        plt.plot(t,x[:,0])
        plt.plot(t, y_DS,'blue')
        plt.plot(t, y_RSS,'green')
        plt.plot(t, y_LSS,'orange')

        plt.grid('on')
        plt.ylabel('$acc_x$')



        y_LSS = np.copy(x[:,1])
        y_LSS[~x_LSS] = np.nan


        y_RSS = np.copy(x[:,1])
        y_RSS[~x_RSS] = np.nan

        
        y_DS = np.copy(x[:,1])
        y_DS[~x_DS] = np.nan

        plt.subplot(312)
        plt.plot(t,x[:,1])
        plt.plot(t, y_DS,'blue')
        plt.plot(t, y_RSS,'green')
        plt.plot(t, y_LSS,'orange')

        plt.grid('on')
        plt.ylabel('$acc_y$')



        y_LSS = np.copy(x[:,2])
        y_LSS[~x_LSS] = np.nan


        y_RSS = np.copy(x[:,2])
        y_RSS[~x_RSS] = np.nan

        
        y_DS = np.copy(x[:,2])
        y_DS[~x_DS] = np.nan

        plt.subplot(313)
        plt.plot(t,x[:,2])
        plt.plot(t, y_DS,'blue')
        plt.plot(t, y_RSS,'green')
        plt.plot(t, y_LSS,'orange')
        plt.ylabel('$acc_z$')
        plt.xlabel('$samples$')
        plt.grid('on')
        plt.show()


    def plot_results(self,X, Y_, means, covariances, title):
        fig = plt.figure()
        splot = plt.subplot(1, 1, 1)

        
        if(covariances is not None):
            for i, (mean, covar, color) in enumerate(zip(
                    means, covariances, color_iter)):
                covar = np.array(covar) 

                if(covar.size != 4):
                    covar_ = covar
                    covar = np.diag(covar_)


                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])
                # as the DP will not use every component it has access to
                # unless it needs it, we shouldn't plot the redundant
                # components.
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='black',linestyle='dashed',linewidth='1.5',ec='black')
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                ell.set_fill(False)
                splot.add_artist(ell)


        else:
            for i, (mean, color) in enumerate(zip(
                    means, color_iter)):
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1],s=0.1, cmap=my_cmap)


        plt.scatter(X[Y_ == 2, 0], X[Y_ == 2, 1],s=0.15, color = [(253.0/255.0,127.0/255.0,14.0/255.0)]) #NAO 0.025
        plt.scatter(X[Y_ == 1, 0], X[Y_ == 1, 1],s=0.15, color = [(44.0/255.0,160.0/255.0,44.0/255.0)]) #NAO 0.025
        plt.scatter(X[Y_ == 0, 0], X[Y_ == 0, 1],s=0.15, color = [(28.0/255.0,117.0/255.0,179.0/255.0)]) #NAO 0.025
        plt.scatter(means[:, 0], means[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='red', zorder=10)
        plt.title(title)
        plt.grid('on')
        plt.show()



    def plot_latent_space(self,g):
        plt.scatter(g.reduced_data_train[:,0],g.reduced_data_train[:,1],.8)
        if(g.pca_dim):
            plt.title(" ")
        else:
            plt.title(" ")
        plt.grid('on')
        plt.show()


    def plot_confusion_matrix(self,cm, classes,
                              title,
                              normalize=True,
                              cmap=plt.cm.Blues):


        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, [classes[1],classes[1],classes[2]], rotation=45)
        plt.yticks(tick_marks, [classes[2],classes[1],classes[0]], rotation=45)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        cm_linesum=np.sum(cm.round(2),axis=1)
        diff_cm=1-cm_linesum
        add_cm=np.zeros_like(cm)+np.diag(diff_cm)
        cm=cm+add_cm
        #        print cm_linesum
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()





class GEM2_data:
    def __init__(self):
        self.lfX = 0
        self.lfY = 0
        self.lfZ = 0
        self.ltX = 0
        self.ltY = 0
        self.ltZ = 0
        self.rfX = 0
        self.rfY = 0
        self.rfZ = 0
        self.rtX = 0
        self.rtY = 0
        self.rtZ = 0
        self.accX = 0
        self.accY = 0
        self.accZ = 0
        self.gX = 0
        self.gY = 0
        self.gZ = 0
        #self.laccX = 0
        #self.laccY = 0
        #self.laccZ = 0
        #self.lgX = 0
        #self.lgY = 0
        #self.lgZ = 0
        #self.raccX = 0
        #self.raccY = 0
        #self.raccZ = 0
        #self.rgX = 0
        #self.rgY = 0
        #self.rgZ = 0
        self.dcX = 0
        self.dcY = 0
        self.dcZ = 0
        self.lvX = 0
        self.lvY = 0
        self.lvZ = 0
        self.lwX = 0
        self.lwY = 0
        self.lwZ = 0
        self.rvX = 0
        self.rvY = 0
        self.rvZ = 0
        self.rwX = 0
        self.rwY = 0
        self.rwZ = 0