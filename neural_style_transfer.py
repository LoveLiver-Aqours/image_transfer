FLAG_JUPYTER = True
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers import Input, Lambda, merge
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.utils.data_utils import get_file
from keras.applications import vgg16
from keras.preprocessing import image

import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b

import IPython.display as IPd

def preprocess_forvgg(img, width, height):
    img = img.resize((width, height))
    img = image.img_to_array(img).astype('float32')
    #print(img.shape)
    img = np.expand_dims(img, axis=0)
    #print(img.shape)
    img = vgg16.preprocess_input(img)

    # 'RGB'->'BGR'
    #img = img[:, :, ::-1]
    # Remove zero-center by mean pixel
    #img[:, :, 0] -= 103.939
    #img[:, :, 1] -= 116.779
    #img[:, :, 2] -= 123.68

    #img = np.expand_dims(img, axis=0)
    print(img.shape)
    return img

def deprocess_image(img, width, height):
    img = img.reshape((height, width, 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def content_loss(base_features, outputs_features):
    return K.sum(K.square(base_features - outputs_features))

def style_loss(style_features, outputs_features):
    A = gram_matrix(style_features)
    G = gram_matrix(outputs_features)
    #N = K.ndim(style_features)
    #print(style_features.shape)
    N = int(style_features.shape[2])
    M = int(style_features.shape[0] * style_features.shape[1])
    return K.sum(K.square(A - G)) / (4. * (N ** 2) * (M ** 2))

# /*--- for smoothing ---*/
def total_loss(img_out, weights_total):
    #print(img_out.shape[2])
    #print(img_out.shape[1])
    a = K.square(img_out[:, :img_out.shape[1] - 1, :img_out.shape[2] - 1, :] - img_out[:, 1:, :img_out.shape[2] - 1, :])
    b = K.square(img_out[:, :img_out.shape[1] - 1, :img_out.shape[2] - 1, :] - img_out[:, :img_out.shape[1] - 1, 1:, :])
    return weights_total * K.sum(K.pow(a + b, 1.25))

def gram_matrix(mat, s=-1):
    features = K.batch_flatten(K.permute_dimensions(mat, (2, 0, 1))) # (channel, height, width) -> flatten: 1 dim vector
    return K.dot(features + s, K.transpose(features + s))

def eval_loss_and_grads(input_img, width, height):
    #width = args[0]
    #height = args[1]
    x = input_img.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x, *args):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, args[0], args[1])
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x, *args):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def minimize_func(input_img, *args):
    width = args[0]
    height = args[1]
    input_img = input_img.reshape((1, height, width, 3))
    outs = f_outputs([input_img])
    return outs[0]

def grads_func(input_img, *args):
    width = args[0]
    height = args[1]
    input_img = input_img.reshape((1, height, width, 3))
    outs = f_outputs([input_img])
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return grad_values

if __name__=='__main__':
    if(not FLAG_JUPYTER):
        parser = argparse.ArgumentParser(description='Style transfer model')
        parser.add_argument('--style', '-s', type=str, required=True,
                            help='style image file name')
        parser.add_argument('--base', '-b', type=str, required=True,
                            help='base image file name')
        parser.add_argument('--output', '-o', default='output', type=str,
                            help='output model file path')
        args = parser.parse_args()
        style_reference_image_path = args.style
        base_image_path = args.base
        output_image_path = args.output
    else:
        style_reference_image_path = 'starry_night.jpg'
        base_image_path = 'SI_80023376_32549.jpg'
        output_image_path = 'output'

    weights_content = 0.005
    weights_style = 1.0
    weights_total = 1e-3

    iter = 10
    print('Settings: ' + K.image_data_format())

    base_img = image.load_img(base_image_path)
    #base_img = imread(base_image_path)
    style_img = image.load_img(style_reference_image_path)
    #style_img = imread(style_reference_image_path)
    #style_img = style_img.crop((0, 0, 300, 500))
    #style_img.show()
    #style_img.save('temp.png')

    output_img_height = 400
    output_img_width = int(base_img.size[0] * output_img_height / base_img.size[1])
    #output_img_width = 400

    _input_base_image = K.variable(preprocess_forvgg(base_img, output_img_width, output_img_height))
    _input_style_image = K.variable(preprocess_forvgg(style_img, output_img_width, output_img_height))
    _output_image = K.placeholder(shape = (1, output_img_height, output_img_width, 3))

    input_tensor = K.concatenate([_input_base_image,
                                  _input_style_image,
                                  _output_image], axis = 0)
    #model2 = vgg16.VGG16(input_tensor=input_tensor,
    #                    weights='imagenet', include_top=False)
    _vgg_input = Input(tensor = input_tensor, batch_shape=(3, output_img_height, output_img_width, 3))
    _x = Convolution2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(_vgg_input)
    _x = Convolution2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(_x)
    _x = AveragePooling2D((2, 2), strides=(2, 2))(_x)
    #_x = MaxPooling2D((2, 2), strides=(2, 2))(_x)

    _x = Convolution2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(_x)
    _x = Convolution2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(_x)
    _x = AveragePooling2D((2, 2), strides=(2, 2))(_x)
    #_x = MaxPooling2D((2, 2), strides=(2, 2))(_x)

    _x = Convolution2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(_x)
    _x = Convolution2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(_x)
    _x = Convolution2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(_x)
    _x = AveragePooling2D((2, 2), strides=(2, 2))(_x)
    #_x = MaxPooling2D((2, 2), strides=(2, 2))(_x)

    _x = Convolution2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(_x)
    _x = Convolution2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(_x)
    _x = Convolution2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(_x)
    _x = AveragePooling2D((2, 2), strides=(2, 2))(_x)
    #_x = MaxPooling2D((2, 2), strides=(2, 2))(_x)

    _x = Convolution2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(_x)
    _x = Convolution2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(_x)
    _x = Convolution2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(_x)
    _x = AveragePooling2D((2, 2), strides=(2, 2))(_x)
    #_x = MaxPooling2D((2, 2), strides=(2, 2))(_x)

    model = Model(input_tensor, _x)
    vgg_weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

    model.load_weights(vgg_weights)
    model.summary()
    #model2.summary()

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    #print(outputs_dict.keys())

    # /*--- calc content loss ---*/
    temp_features = outputs_dict['conv5_2']
    #temp_features = outputs_dict['block5_conv2']
    temp_base_img_features = temp_features[0, :, :, :]
    temp_output_img_featurs = temp_features[2, :, :, :]

    c_loss = weights_content * content_loss(temp_base_img_features, temp_output_img_featurs)

    # /*--- calc style loss ---*/
    target_layers = ['conv1_1', 'conv1_2',
                     'conv2_1', 'conv2_2',
                     'conv3_1', 'conv3_2', 'conv3_3',
                     'conv4_1', 'conv4_2', 'conv4_3',
                     'conv5_1', 'conv5_2', 'conv5_3']
    #target_layers = ['block1_conv1', 'block2_conv1',
    #                 'block3_conv1', 'block4_conv1',
    #                 'block5_conv1']

    s_loss = K.variable(0.)
    for t_layers in target_layers:
        temp_features = outputs_dict[t_layers]
        temp_style_img_features = temp_features[1, :, :, :]
        temp_output_img_featurs = temp_features[2, :, :, :]
        #print(temp_style_img_features.shape)
        #print(temp_output_img_featurs.shape)
        temp_style_loss = style_loss(temp_style_img_features, temp_output_img_featurs)
        s_loss = s_loss + (weights_style / len(target_layers)) * temp_style_loss

    loss = c_loss + s_loss + total_loss(_output_image, weights_total)
    grads = K.gradients(loss, _output_image)
    _outputs = [loss]
    _outputs += grads

    f_outputs = K.function([_output_image], _outputs)

    #init_img = np.random.uniform(0, 255, (1, output_img_height, output_img_width, 3))

    #init_img[0, :, :, 0] -= 103.939
    #init_img[0, :, :, 1] -= 116.779
    #init_img[0, :, :, 2] -= 123.68

    #init_img = vgg16.preprocess_input(init_img)

    #print(init_img.shape)
    init_img = preprocess_forvgg(base_img, output_img_width, output_img_height) # change noise image
    #print(input_base_img.shape)
    evaluator = Evaluator()
    loss_list = []

    for i in range(iter):
        print('Start of iteration', i)
        start_time = time.time()
        init_img, min_val, info = fmin_l_bfgs_b(evaluator.loss, init_img.flatten(), args = (output_img_width, output_img_height), fprime=evaluator.grads, maxfun=20)
        loss_list.append(min_val)
        print('Current loss value:', min_val)
        # save current generated image
        img = deprocess_image(init_img.copy(), output_img_width, output_img_height)
        fname = output_image_path + '_at_iteration_%02d.png' % i
        if(not FLAG_JUPYTER):
            image.save_img(fname, img)
        else:
            pilImg = Image.fromarray(np.uint8(img))
            pilImg.save(fname, 'PNG', quality=100, optimize=True)
            IPd.display_png(IPd.Image(fname))
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    if(FLAG_JUPYTER):
        %matplotlib inline
        plt.plot(range(0, iter), loss_list)
