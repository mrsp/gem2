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


from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np

def clf_loss(y_true, y_pred):
    #x  = 1.0 * K.square(y_true[:,6] - (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]))
    #y  = 1.0 * K.square(y_true[:,7] - (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]))
    #z  = 1.0 * K.square(y_true[:,8] - (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]))
    #loss = K.mean(K.sqrt(x + y + z + K.epsilon()))


    x  = 1.0 * K.abs(y_true[:,6] - (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]))
    y  = 1.0 * K.abs(y_true[:,7] - (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]))
    z  = 1.0 * K.abs(y_true[:,8] - (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]))
    #loss = K.sum(x + y + z, axis = -1)
    loss = K.mean(x + y + z)
    return loss

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)) 


class supervisedVariationalAutoencoder():
    def __init__(self):
        self.firstrun = True
    
    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(self,args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def setDimReduction(self, input_dim, latent_dim, intermediate_dim, num_classes):
        # VAE model = encoder + decoder
        vae_loss_weight = 1.0
        # build encoder model
        inputs = Input(shape=(input_dim,), name='encoder_input')
        x = Dense(intermediate_dim, activation='swish', use_bias = False)(inputs)
        z_mean = Dense(latent_dim, activation='swish',  use_bias = False, name='z_mean')(x)
        z_log_var = Dense(latent_dim, activation='swish', use_bias = False, name='z_log_var')(x)
        # use reparameterization trick to push the sampling out as input
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        #self.encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        self.predicted = Dense(latent_dim, activation='softmax', name='class_output', use_bias=True)(z)


        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        decoded = Dense(intermediate_dim, activation='swish', use_bias = False)(latent_inputs)
        decoded = Dense(input_dim, activation='swish', use_bias = False)(decoded)
        # instantiate decoder model
        decoder = Model(latent_inputs, decoded, name='reconst_output')
        #decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        # New: Add another output for classification
        outputs = [decoder(self.encoder(inputs)[2]), self.predicted]
        self.model = Model(inputs, outputs, name='vae_mlp')
        #self.model.summary()
        reconstruction_loss = mae(inputs, outputs[0])
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = vae_loss_weight * K.mean(reconstruction_loss + kl_loss)
        self.model.add_loss(vae_loss)

        # New: add the clf loss
        self.model.compile(optimizer='adam', loss={'class_output': clf_loss},loss_weights={'class_output': 0.1})
        #self.model.summary()
        #plot_model(self.model, to_file='supervised_vae.png', show_shapes=True)

    def fit(self, x_train, y_train, x_validation, y_validation, epochs, batch_size):
        # reconstruction_loss = binary_crossentropy(inputs, outputs)
        self.model_log = self.model.fit(x_train, {'reconst_output':x_train, 'class_output': y_train}, validation_data = (x_validation, {'reconst_output':x_validation, 'class_output': y_validation}), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
