from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Lambda, merge
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing import image

import numpy as np
import time
import argparse
from PIL import Image

from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b

def preprocess_forvgg(img, width, height):
    img = img.resize((width, height))
    img = image.img_to_array(img)
    #print(img.shape)
    img = np.expand_dims(img, axis=0)
    #print(img.shape)
    img = vgg16.preprocess_input(img)
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

# /*--- ??? ---*/
def total_loss(img_out, weights_total):
    print(img_out.shape[2])
    print(img_out.shape[1])
    a = K.square(img_out[:, :img_out.shape[1] - 1, :img_out.shape[2] - 1, :] - img_out[:, 1:, :img_out.shape[2] - 1, :])
    b = K.square(img_out[:, :img_out.shape[1] - 1, :img_out.shape[2] - 1, :] - img_out[:, :img_out.shape[1] - 1, 1:, :])
    return weights_total * K.sum(K.pow(a + b, 1.25))

def gram_matrix(mat):
    features = K.batch_flatten(K.permute_dimensions(mat, (2, 0, 1))) # (channel, height, width) -> flatten: 1 dim vector
    return K.dot(features, K.transpose(features))

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
    parser = argparse.ArgumentParser(description='Style transfer model')
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name')
    parser.add_argument('--base', '-b', type=str, required=True,
                        help='base image file name')
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file path')
    args = parser.parse_args()
    weights_alpha = 0.025
    weights_beta = 1.0
    iter = 10
    print('Settings: ' + K.image_data_format())

    style_reference_image_path = args.style
    base_image_path = args.base

    base_img = image.load_img(base_image_path)
    style_img = image.load_img(style_reference_image_path)
    style_img = style_img.crop((0, 0, 300, 500))
    #style_img.show()
    #style_img.save('temp.png')

    output_img_height = 400
    output_img_width = int(base_img.size[0] * output_img_height / base_img.size[1])

    _input_base_image = K.variable(preprocess_forvgg(base_img, output_img_width, output_img_height))
    _input_style_image = K.variable(preprocess_forvgg(style_img, output_img_width, output_img_height))
    _output_image = K.placeholder(shape = (1, output_img_height, output_img_width, 3))

    input_tensor = K.concatenate([_input_base_image,
                                  _input_style_image,
                                  _output_image], axis = 0)
    model = vgg16.VGG16(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    #print(outputs_dict.keys())

    # /*--- calc content loss ---*/
    temp_features = outputs_dict['block5_conv2']
    temp_base_img_features = temp_features[0, :, :, :]
    temp_output_img_featurs = temp_features[2, :, :, :]

    c_loss = weights_alpha * content_loss(temp_base_img_features, temp_output_img_featurs)

    # /*--- calc style loss ---*/
    target_layers = ['block1_conv1', 'block2_conv1',
                     'block3_conv1', 'block4_conv1',
                     'block5_conv1']

    s_loss = K.variable(0.)
    for t_layers in target_layers:
        temp_features = outputs_dict[t_layers]
        temp_style_img_features = temp_features[1, :, :, :]
        temp_output_img_featurs = temp_features[2, :, :, :]
        #print(temp_style_img_features.shape)
        #print(temp_output_img_featurs.shape)
        temp_style_loss = style_loss(temp_style_img_features, temp_output_img_featurs)
        s_loss = s_loss + (weights_beta / len(target_layers)) * temp_style_loss

    loss = c_loss + s_loss + total_loss(_output_image, 1.0)
    grads = K.gradients(loss, _output_image)
    _outputs = [loss]
    _outputs += grads

    f_outputs = K.function([_output_image], _outputs)

    input_base_img = preprocess_forvgg(base_img, output_img_width, output_img_height) # change noise image
    evaluator = Evaluator()

    for i in range(iter):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, input_base_img.flatten(), args = (output_img_width, output_img_height), fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # save current generated image
        img = deprocess_image(x.copy(), output_img_width, output_img_height)
        fname = 'test' + '_at_iteration_%d.png' % i
        image.save_img(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))