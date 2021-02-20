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
from tensorflow.keras.layers import Input, Dense
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

class supervisedAutoencoder():
    def __init__(self):
        self.firstrun = True
        make_keras_picklable()


    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


    def mae(self, y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true))


    def logcosh(self, y_true, y_pred):
        return tf.math.log((tf.math.exp(y_pred - y_true) + tf.math.exp(-(y_pred - y_true)))/2)


    def clf_loss(self, y_true, y_pred):
      
        '''
        wx  = 0.0 * (y_true[:,6] - tf.math.divide( (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]),(y_pred[:,0]+y_pred[:,1]) ))
        wy  = 1.0 * (y_true[:,7] - tf.math.divide( (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]),(y_pred[:,0]+y_pred[:,1]) ))
        wz  = 1.0 * (y_true[:,8] - tf.math.divide( (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]),(y_pred[:,0]+y_pred[:,1]) ))
        ax  = 0.1 * (y_true[:,15] - tf.math.divide( (y_pred[:,0]*y_true[:,9] + y_pred[:,1]*y_true[:,12]),  (y_pred[:,0]+y_pred[:,1]) ))
        ay  = 0.0 * (y_true[:,16] - tf.math.divide( (y_pred[:,0]*y_true[:,10] + y_pred[:,1]*y_true[:,13]), (y_pred[:,0]+y_pred[:,1]) ))
        az  = 0.0 * (y_true[:,17] - tf.math.divide( (y_pred[:,0]*y_true[:,11] + y_pred[:,1]*y_true[:,14]), (y_pred[:,0]+y_pred[:,1]) ))
        tmp = wx + wy + wz + ax + ay + az
        loss = K.sum(tf.math.log((tf.math.exp(tmp) + tf.math.exp(-tmp))/2 + 0.01), axis = -1)
        '''
        '''
        wx  = 0.0 * K.abs(y_true[:,6] - tf.math.divide( (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3]),(y_pred[:,0]+y_pred[:,1]) ))
        wy  = 1.0 * K.abs(y_true[:,7] - tf.math.divide( (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4]),(y_pred[:,0]+y_pred[:,1]) ))
        wz  = 1.0 * K.abs(y_true[:,8] - tf.math.divide( (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5]),(y_pred[:,0]+y_pred[:,1]) ))


        ax  = 0.1 * K.abs(y_true[:,15] - tf.math.divide( (y_pred[:,0]*y_true[:,9] + y_pred[:,1]*y_true[:,12]),  (y_pred[:,0]+y_pred[:,1]) ))
        ay  = 0.0 * K.abs(y_true[:,16] - tf.math.divide( (y_pred[:,0]*y_true[:,10] + y_pred[:,1]*y_true[:,13]), (y_pred[:,0]+y_pred[:,1]) ))
        az  = 0.0 * K.abs(y_true[:,17] - tf.math.divide( (y_pred[:,0]*y_true[:,11] + y_pred[:,1]*y_true[:,14]), (y_pred[:,0]+y_pred[:,1]) ))
    
        '''

        wx = 1.0 * K.abs(y_true[:, 6] - ((y_pred[:, 0]*y_true[:, 0] +
                                        y_pred[:, 1]*y_true[:, 3])))
        wy = 1.0 * K.abs(y_true[:, 7] - ((y_pred[:, 0]*y_true[:, 1] +
                                        y_pred[:, 1]*y_true[:, 4])))
        wz = 1.0 * K.abs(y_true[:, 8] - ((y_pred[:, 0]*y_true[:, 2] +
                                        y_pred[:, 1]*y_true[:, 5])))

        ax = 1.0 * K.abs(y_true[:, 15] - ((y_pred[:, 0]*y_true[:, 9] +
                                            y_pred[:, 1]*y_true[:, 12]) ))
        ay = 1.0 * K.abs(y_true[:, 16] - ((y_pred[:, 0]*y_true[:, 10] +
                                        y_pred[:, 1]*y_true[:, 13]) ))
        az = 1.0 * K.abs(y_true[:, 17] - ((y_pred[:, 0]*y_true[:, 11] +
                                        y_pred[:, 1]*y_true[:, 14]) ))

        '''
        
        wx  = 0.0 * K.abs(y_true[:,6] -( (y_pred[:,0]*y_true[:,0] + y_pred[:,1]*y_true[:,3])/(y_pred[:,0]+y_pred[:,1]) ))
        wy  = 1.0 * K.abs(y_true[:,7] -( (y_pred[:,0]*y_true[:,1] + y_pred[:,1]*y_true[:,4])/(y_pred[:,0]+y_pred[:,1]) ))
        wz  = 1.0 * K.abs(y_true[:,8] -( (y_pred[:,0]*y_true[:,2] + y_pred[:,1]*y_true[:,5])/(y_pred[:,0]+y_pred[:,1]) ))


        ax  = 0.1 * K.abs(y_true[:,15] - ( (y_pred[:,0]*y_true[:,9] + y_pred[:,1]*y_true[:,12])/  (y_pred[:,0]+y_pred[:,1]) ))
        ay  = 0.1 * K.abs(y_true[:,16] - ( (y_pred[:,0]*y_true[:,10] + y_pred[:,1]*y_true[:,13])/ (y_pred[:,0]+y_pred[:,1]) ))
        az  = 0.0 * K.abs(y_true[:,17] - ( (y_pred[:,0]*y_true[:,11] + y_pred[:,1]*y_true[:,14])/ (y_pred[:,0]+y_pred[:,1]) ))
        '''

        loss = K.mean(wx + wy + wz) + K.mean(ax + ay + az)

        return loss

    def setDimReduction(self, input_dim, latent_dim, intermediate_dim, num_classes):
        sae_input = Input(shape=(input_dim,), name='input')
        # this model maps an input to its encoded representation
        #encoded = Dense(15, activation='tanh', name='encode_0')(sae_input)
        encoded = Dense(intermediate_dim, activation='tanh',
                        name='encode_1')(sae_input)
        #encoded = Dense(2, activation='selu', name='encode_2')(encoded)
        initializer = tf.keras.initializers.Ones()
        predicted = Dense(2, activation='sigmoid', name='class_output',
                        kernel_initializer=initializer)(encoded)
        encoded = predicted
        #predicted = encoded
        self.encoder = Model(sae_input, predicted)
        # Reconstruction Decoder: Latent to input
        decoded = Dense(intermediate_dim, activation='tanh',
                        name='decode_1')(predicted)
        #decoded = Dense(15, activation='tanh', name='decode_2')(decoded)
        decoded = Dense(input_dim, activation='tanh',
                        name='reconst_output')(decoded)
        # Take input and give classification and reconstruction
        self.model = Model(inputs=[sae_input], outputs=[
                           decoded, encoded, predicted])
        self.model.compile(optimizer='rmsprop',
                           loss={'class_output': self.clf_loss,
                                 'reconst_output': self.logcosh},
                           loss_weights={'class_output': 0.5,
                                         'reconst_output': 1.0})
        # self.model.summary()
        self.firstrun = False

    def fit(self, x_train, y_train, x_validation, y_validation, epochs_, batch_size_):
        self.model_log = self.model.fit(x_train, {'reconst_output': x_train, 'class_output': y_train}, validation_data=(x_validation, {
                                        'reconst_output': x_validation, 'class_output': y_validation}), epochs=epochs_, batch_size=batch_size_,  verbose=1, shuffle=True)
