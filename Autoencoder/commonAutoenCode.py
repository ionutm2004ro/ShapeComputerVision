def load_data(debug=False, edgeDetect=False, folder='Fill2NoBg'):
    from tensorflow.keras.utils import to_categorical
    
    # loading data
    PATH = os.getcwd()
    # Define data path
    data_path = PATH + '/../ShapeData/' + folder
    data_dir_list = os.listdir(data_path)
    data_dir_list = [data_dir_list[2]] #0=Ellipse 1=Quadrilateral 2=Triangle
    img_data_list = []


    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            if edgeDetect:
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img,1)
                img_data_list.append(edge_detection(input_img))
            else:
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img,0)#1=RGB,0=Grey,-1=unchanged
                img_data_list.append(input_img)
#            input_img_resize=cv2.resize(input_img,(IMAGE_HEIGHT,IMAGE_WIDTH))
#            img_data_list.append(input_img_resize)
#             img_data_list.append(edge_detection(input_img))
    
    X_train = np.array(img_data_list)
    
    #Reshaping
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], num_channels))
    
    if debug:
        print('After preprocessing: ')
        print(' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
        print(' - X_val.shape = {}, y_val.shape = {}'.format(X_val.shape, y_val.shape))
        print(' - X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))
        print(' - test_images.shape = {}, test_labels.shape = {}'.format(
            test_images.shape, test_labels.shape))  
    
    return X_train
#########################################################################################################################
import numpy as np
import os,cv2
X = load_data()
X = X.astype('float32') / 255.0 - 0.5
encoding_dim = 100
#########################################################################################################################
import matplotlib.pyplot as plt
def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1),cmap='gray')
#########################################################################################################################
#https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-ii-24b9cca69bd6
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
import tensorflow as tf

class UncorrelatedFeaturesConstraint (Constraint):
    
    def __init__(self, encoding_dim, weightage = 1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
    
    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))
        
        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)
        
        return covariance
            
    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            output = K.sum(K.square(self.covariance - K.dot(self.covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)
#########################################################################################################################
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations,initializers,regularizers,constraints
from tensorflow.keras.layers import InputSpec
class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
#########################################################################################################################
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
#     encoder.add(Dense(code_size))
    encoder.add(Dense(encoding_dim,
                      activation="linear",
                      input_shape = (img_shape,),
                      use_bias = True,
                      activity_regularizer = UncorrelatedFeaturesConstraint(encoding_dim, weightage = 1.)))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
#     decoder.add(Dense(np.prod(img_shape))) 
    decoder.add(Dense(np.prod(img_shape), activation="linear", use_bias = True))
#     decoder.add(DenseTied(np.prod(img_shape), activation="linear", use_bias = False, tied_to = encoder))
        # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder
#########################################################################################################################
IMG_SHAPE = X.shape[1:]
encoder, decoder = build_autoencoder(IMG_SHAPE, encoding_dim)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(metrics=['accuracy'],optimizer='adamax', loss='mse')
# autoencoder.compile(metrics=['accuracy'],optimizer='sgd', loss='mse')

print(autoencoder.summary())




# input_dim = X.shape[1:]
# encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True, activity_regularizer=UncorrelatedFeaturesConstraint(encoding_dim, weightage = 1.)) 
# decoder = Dense(np.prod(input_dim), activation="linear", use_bias = True)

# autoencoder = Sequential()
# autoencoder.add(encoder)
# autoencoder.add(decoder)

# autoencoder.compile(metrics=['accuracy'],
#                     loss='mean_squared_error',
#                     optimizer='sgd')
# print(autoencoder.summary())