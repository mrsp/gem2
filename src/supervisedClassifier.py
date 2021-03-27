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

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model,load_model, save_model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
import tensorflow as tf

import tempfile
import os
# Hotfix function
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False, dir=os.getcwd()) as fd:
            save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        os.unlink(fd.name)
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False, dir=os.getcwd()) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
        os.unlink(fd.name)
        self.__dict__ = model.__dict__


    cls = Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__



class supervisedClassifier():
    def __init__(self):
        self.firstrun = True
        make_keras_picklable()



    def LLeg_loss(self, y_true, y_pred):
        wx = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 0])
        wy = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 1])
        wz = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 2])

        ax = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 9])
        ay = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 10])
        az = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 11])
        loss = K.mean(ax*ay*az*wx*wy*wz)

        return loss

    def RLeg_loss(self, y_true, y_pred):
        wx = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 3])
        wy = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 4])
        wz = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 5])

        ax = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 12])
        ay = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 13])
        az = 1.0 * K.abs(y_pred[:, 0] - y_true[:, 14])
        loss = K.mean(ax*ay*az*wx*wy*wz)
        return loss


    def setDimensions(self, input_dim, latent_dim, intermediate_dim):
        sc_input = Input(shape=(input_dim,), name='input')

        #left_leg_input = Lambda(lambda x: x[:,0:12])(sc_input)
        #right_leg_input = Lambda(lambda x: x[:,6:])(sc_input)

        #h1_out = Dense(6, activation='selu')(left_leg_input)  # only connected to the second neuron
        #h2_out = Dense(6, activation='selu')(right_leg_input)  # connected to both neurons

        #initializer = tf.keras.initializers.Constant(0.5)

        #outL = Dense(1, activation='sigmoid', name='LLeg_out', kernel_initializer=initializer, use_bias=False)(h1_out)
        #outR = Dense(1, activation='sigmoid', name='RLeg_out', kernel_initializer=initializer, use_bias=False)(h2_out)
        #output = concatenate([outL,outR])
        

         
        hout= Dense(6, activation='sigmoid')(sc_input)  

        initializer = tf.keras.initializers.Constant(0.5)
        output = Dense(2, activation='sigmoid', kernel_initializer=initializer, use_bias=True)(hout)
        self.model = Model(sc_input, output)
        # Compile the model
        self.model.compile(optimizer='adam', loss="mean_squared_logarithmic_error")
        #self.model.summary()
        self.firstrun = False


    def fit(self, x_train, y_train, x_validation, y_validation, epochs, batch_size):
        self.model_log = self.model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=epochs, batch_size=batch_size,  verbose=1, shuffle=True)

