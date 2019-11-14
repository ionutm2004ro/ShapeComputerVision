#import warnings
#warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True

import os, random, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

float_formatter = lambda x: '%.4f' % x
np.set_printoptions(formatter={'float_kind':float_formatter})
np.set_printoptions(threshold=np.inf, suppress=True, precision=4)

plt.style.use("seaborn-colorblind")
plt.rcParams["figure.figsize"] = (8, 6)
sns.set_style("darkgrid")
#sns.set_context("talk")
sns.set_context(context='notebook', font_scale=1.25)
sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

# NOTE: It is important that you set a seed value to get same results in every run.
# Any number is Ok.
seed = 817#123
random.seed(seed)
np.random.seed(seed)

#class_names = ['Ellipse','4lateral','3angle']
class_names = ['0','4','3']
num_channels, KERNEL_SIZE, alpha = 1, (9,9), 0.0003
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SHAPE, num_classes = 40, 40, (40,40,num_channels), len(class_names)
# training parameters
NUM_EPOCHS, BATCH_SIZE = 15, 128
# for loading & saving model state
KR_MODEL_NAME = 'keras_shapes.h5'
#########################################################################################################################
def edge_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    return cv2.Canny(frame,70,120)
#########################################################################################################################
def load_and_preprocess_data(debug=False, edgeDetect=False, folder='Fill2NoBg'):
    from keras.utils import to_categorical
    
    # loading data
    PATH = os.getcwd()
    # Define data path
    data_path = PATH + '/../ShapeData/' + folder
    data_dir_list = os.listdir(data_path)

    img_data_list=[]

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
    num_data = X_train.shape[0]
    
    # Assigning Labels
    y_train = np.ones((num_data,),dtype='int64')
    for i in range(0,num_classes):
        y_train[int(np.floor((i)*num_data/num_classes)):int(np.floor((i+1)*num_data/num_classes))]=i
    
    if debug:
        print('Before preprocessing:')
        print(' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))

    #shuffle data
    indexes = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[indexes], y_train[indexes]
    
    #split data into 3 parts & keep copy of test images and labels
    num_train=int(np.floor(num_data*0.8))
    num_cross=int(np.floor(num_train*0.2))
    
    X_val, y_val = X_train[num_train:num_train+num_cross], y_train[num_train:num_train+num_cross]
    X_test, y_test = X_train[num_train+num_cross:], y_train[num_train+num_cross:]
    test_images, test_labels = X_test.copy(), y_test.copy()
    X_train, y_train = X_train[:num_train], y_train[:num_train]
    
    #scale images and one-hot encode labels(from 0 to [1,0,0] ,1 to [0,1,0] etc)
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes) 
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    #Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], num_channels))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], num_channels))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], num_channels))
    
    if debug:
        print('After preprocessing: ')
        print(' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
        print(' - X_val.shape = {}, y_val.shape = {}'.format(X_val.shape, y_val.shape))
        print(' - X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))
        print(' - test_images.shape = {}, test_labels.shape = {}'.format(
            test_images.shape, test_labels.shape))  
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (test_images, test_labels)
#########################################################################################################################
def display_sample(sample_images, sample_labels, sample_predictions=None,
                   num_rows=5, num_cols=10, plot_title=None, fig_size=None):
    """ 
    display a random selection of images & corresponding labels, optionally with predictions 
    The display is laid out in a grid of num_rows x num_col cells.
    If sample_predictions are provided, then each cell's title displays the prediction (if it 
    matches actual value) else it displays actual value/prediction. 
    """
    assert sample_images.shape[0] == num_rows * num_cols
    
    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows,num_cols,figsize=((14, 9) if fig_size is None else fig_size),
            gridspec_kw={"wspace": 0.02, "hspace": 0.30}, squeeze=True)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                ax[r, c].imshow(sample_images[image_index] 
                                ,cmap="gray" #cmap grey so the image is black and white, not 2 random colours
                                ,vmin=0,vmax=255 #vmin and vmax is needed otherwise the image will be two tone
                                )

                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title(class_names[sample_labels[image_index]])
                else:
                    # else check if prediction matches actual value
                    true_label = class_names[sample_labels[image_index]]
                    pred_label = class_names[sample_predictions[image_index]]
                    prediction_matches_true = \
                        (sample_labels[image_index] == sample_predictions[image_index])
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = true_label
                        title_color = 'g'
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        #title = 'Num: %s/%s' % (true_label, pred_label)
                        title = pred_label +'/'+ true_label
                        title_color = 'r'
                    # display cell title
                    title = ax[r, c].set_title(title)
                    plt.setp(title, color=title_color)
        # set plot title, if one specified
        if plot_title is not None: f.suptitle(plot_title)
    
        plt.show()
        plt.close()
#########################################################################################################################
(X_train, y_train), (X_val, y_val), (X_test, y_test), (test_images, test_labels) = \
    load_and_preprocess_data(edgeDetect=True)