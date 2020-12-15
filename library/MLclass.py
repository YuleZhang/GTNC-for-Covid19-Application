import copy
import math
import os
import time

import cv2
import numpy
import pandas as pd
import scipy.special
import skimage
import torch
import torchvision
from skimage import io,transform
import numpy as np

from library import Parameters


class MachineLearning:

    def __init__(self, para=Parameters.ml(), debug_mode=False):
        # initialize parameters
        self.para = copy.deepcopy(para)
        self.images_data = dict()
        self.index = dict()
        self.labels_data = dict()
        self.update_info = dict()
        self.data_info = dict()
        self.tmp = {}
        self.generative_model = ['GTN', 'GTNO',  'GTNO_OneLayer',  'GTNO_TwoLayer', 'GJEPG']
        self.discriminative_model = ['DTNC']

    def initialize_dataset(self):
        # Load the data set
        self.load_dataset()
        self.data_info['labels'] = tuple(list(self.labels_data['train']))
        # self.data_info['n_training'] = self.para['n_training']
        # Deal data
        self.calculate_dataset_info()
        # Arrange Data
        self.arrange_data()
        if self.para['classifier_type'] in self.generative_model:
            self.images_data['dealt_input'] = self.deal_data(self.images_data['input'])
            self.data_info['n_training'], self.data_info['n_feature'] = \
                self.images_data['dealt_input'].shape

    def start_learning(self, learning_loops=30):
        self.print_program_info(mode='start')
        if self.update_info['is_converged'] is not True:
            self.prepare_start_learning()
            if self.update_info['is_converged'] == 'untrained':
                self.update_info['is_converged'] = False
            while self.update_info['loops_learned'] >= learning_loops:
                print('you have learnt too many loops')
                learning_loops = int(input("learning_loops = "))
            if self.update_info['loops_learned'] == 0:
                self.calculate_cost_function()
                self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
                print('Initializing ... cost function = ' + str(self.update_info['cost_function']))
            if not self.update_info['is_converged']:
                print('start to learn to ' + str(learning_loops) + ' loops')
            while (self.update_info['loops_learned'] < learning_loops) and not(self.update_info['is_converged']):
                self.update_one_loop()
                self.is_converge()
                self.save_data()
        else:
            print('load converged mps, do not need training.')
        if self.update_info['is_converged']:
            self.print_converge_info()
        else:
            print('Training end, cost function = ' + str(self.update_info['cost_function']) + ', do not converge.')
        self.calculate_program_info_time(mode='end')
        self.print_program_info(mode='end')

    def load_dataset(self):
        if self.para['dataset'] == 'mnist':
            data_tmp = torchvision.datasets.MNIST(root=self.para['path_dataset'], download=True, train=True)
            self.images_data['train'] = data_tmp.data.numpy().reshape(-1, 784)
            self.labels_data['train'] = data_tmp.targets.numpy()
            data_tmp = torchvision.datasets.MNIST(root=self.para['path_dataset'], download=True, train=False)
            self.images_data['test'] = data_tmp.data.numpy().reshape(-1, 784)
            self.labels_data['test'] = data_tmp.targets.numpy()
            self.data_info['origin_shape'] = (28, 28)
            del data_tmp
        elif self.para['dataset'] == 'fashion':
            data_tmp = torchvision.datasets.FashionMNIST(root=self.para['path_dataset'], download=True, train=True)
            self.images_data['train'] = data_tmp.data.numpy().reshape(-1, 784)
            self.labels_data['train'] = data_tmp.targets.numpy()
            data_tmp = torchvision.datasets.FashionMNIST(root=self.para['path_dataset'], download=True, train=False)
            self.images_data['test'] = data_tmp.data.numpy().reshape(-1, 784)
            self.labels_data['test'] = data_tmp.targets.numpy()
            self.data_info['origin_shape'] = (28, 28)
            del data_tmp
        elif self.para['dataset'] == 'Berkeley Segmentation Dataset':
            img_path = self.para['path_dataset'] + 'BSDS300/images/'
            for data_type in self.para['data_type']:
                img_path_tmp = img_path + data_type + '/'
                all_names = os.listdir(img_path_tmp)
                self.images_data[data_type] = list()
                for name in all_names:
                    tmp_img = io.imread(img_path_tmp + name)
                    if tmp_img.shape[0] > tmp_img.shape[1]:
                        tmp_img = numpy.transpose(tmp_img, [1, 0, 2])
                    self.images_data[data_type].append(tmp_img.flatten())
                self.images_data[data_type] = numpy.array(self.images_data[data_type])
                self.labels_data[data_type] = numpy.zeros(self.images_data[data_type].shape[0])
            self.data_info['origin_shape'] = (321, 481, 3)
        elif self.para['dataset'] == 'X-Ray Image DataSet':
            img_info_path = self.para['path_dataset']
            if not os.path.exists(img_info_path):
                self.Restore_Img_Path(self.para['dataset_content'],img_info_path)
            img_info = pd.read_csv(img_info_path)
            train_data, train_label, test_data, test_label = self.split_data(img_info, 0.2)
            self.images_data['train'] = numpy.array(train_data)
            self.labels_data['train'] = numpy.array(train_label)
            self.images_data['test'] = numpy.array(test_data)
            self.labels_data['test'] = numpy.array(test_label)
            self.data_info['origin_shape'] = self.para['resize_size']
        elif self.para['dataset'] == 'extended YaleB':
            img_info_path = self.para['path_dataset']
            if not os.path.exists(img_info_path):
                self.Restore_Yale_Img_Path(self.para['dataset_content'],img_info_path)
            img_info = pd.read_csv(img_info_path)
            #print(img_info)
            train_data, train_label, test_data, test_label = self.split_data(img_info, 0.2)
            self.images_data['train'] = numpy.array(train_data)
            self.labels_data['train'] = numpy.array(train_label)
            self.images_data['test'] = numpy.array(test_data)
            self.labels_data['test'] = numpy.array(test_label)

            self.data_info['origin_shape'] = self.para['resize_size']

        self.data_info['shape'] = copy.copy(self.data_info['origin_shape'])

    # The infomation of dataset including the number of train samples and test samples
    # And the n_feature indicates the total pixels of one example,such as 784 in MNIST
    def calculate_dataset_info(self):
        self.data_info['n_sample'] = dict()
        for data_type in self.para['data_type']:
            self.data_info['n_sample'][data_type] = dict()
            self.data_info['n_sample'][data_type] = (self.images_data[data_type]).shape[0]
        self.data_info['n_feature'] = (self.images_data['train']).shape[1]
        for data_type in self.para['data_type']:
            self.index[data_type] = dict()
            self.index[data_type]['origin'] = numpy.arange(self.data_info['n_sample'][data_type])

    # divide the whole data into ten parts according the diverse labels
    # the data will be load in self.index[data_type]['divided'][label]
    def divide_data(self, data_type='train'):
        self.index[data_type]['divided'] = dict()
        if self.para['divide_module'] == 'label':
            # redundant loops in this transverse, because we just have to tranverse the ten labels. 
            for label in self.data_info['labels']:
                self.index[data_type]['divided'][label] = numpy.where(label == self.labels_data[data_type])[0]

    # Load the images of label(self.para['training_label']), and shuffle it, finally store as images_data['input']
    # Also update the parameter data_info['n_traninging']
    def arrange_data(self):
        if self.para['sort_module'] == 'rand':
            self.images_data['train'], self.labels_data['train'] = \
                self.rand_sort_data(self.images_data['train'], self.labels_data['train'])
        self.divide_data(data_type='train')
        if self.para['classifier_type'] in self.generative_model:
            self.images_data['input'] = list()
            self.labels_data['input'] = list()
            for label in tuple(list(self.para['training_label'])):
                self.images_data['input'] += list((self.images_data['train'][
                                            self.index['train']['divided'][label]]))
                self.labels_data['input'] += list((self.labels_data['train'][
                                        self.index['train']['divided'][label]]))
            self.images_data['input'] = numpy.array(self.images_data['input'])
            self.labels_data['input'] = numpy.array(self.labels_data['input'])
            self.images_data['input'], self.labels_data['input'] = \
                self.rand_sort_data(self.images_data['input'], self.labels_data['input'])
            if (self.para['n_training'] == 'all') or (self.para['n_training'] > len(self.labels_data['input'])):
                self.data_info['n_training'] = len(self.labels_data['input'])
            else:
                self.data_info['n_training'] = self.para['n_training']
            if self.data_info['n_training'] < len(self.labels_data['input']):
                self.images_data['input'] = self.images_data['input'][range(self.data_info['n_training']), :]
                self.labels_data['input'] = self.labels_data['input'][range(self.data_info['n_training'])]

    # Mapping the image_data into Hirbert space according the latex, defalut bond dimention is 2
    def feature_map(self, image_data_mapping):
        image_data_mapping = numpy.array(image_data_mapping)
        if self.para['map_module'] == 'many_body_Hilbert_space':
            image_data_mapping = image_data_mapping * self.para['theta']
            while numpy.ndim(image_data_mapping) < 2:
                image_data_mapping.shape = (1,) + image_data_mapping.shape
            image_data_mapping = torch.tensor(image_data_mapping, device=self.device, dtype=self.dtype)
            image_data_mapped = torch.zeros(
                (image_data_mapping.shape + (self.para['mapped_dimension'],)), device=self.device, dtype=self.dtype)
            for ii in range(self.para['mapped_dimension']):
                image_data_mapped[:, :, ii] = math.sqrt(
                    scipy.special.comb(self.para['mapped_dimension'] - 1, ii)) * (
                        torch.sin(image_data_mapping) ** (self.para['mapped_dimension'] - ii - 1)) * (
                        torch.cos(image_data_mapping) ** ii)
        elif self.para['map_module'] == 'linear_map':
            while numpy.ndim(image_data_mapping) < 2:
                image_data_mapping.shape = (1,) + image_data_mapping.shape
            image_data_mapping = torch.tensor(image_data_mapping, device=self.device, dtype=self.dtype)
            image_data_mapped = torch.stack((image_data_mapping, 1-image_data_mapping), 2)
            if not self.para['mapped_dimension'] == 2:
                print('check you code, mapped_dimension is wrong')
                image_data_mapped = False
        elif self.para['map_module'] == 'sqrt_linear_map':
            while numpy.ndim(image_data_mapping) < 2:
                image_data_mapping.shape = (1,) + image_data_mapping.shape
            image_data_mapping = torch.tensor(image_data_mapping, device=self.device, dtype=self.dtype)
            image_data_mapped = torch.stack((
                image_data_mapping ** 0.5, (1-image_data_mapping) ** 0.5), 2)
            if not self.para['mapped_dimension'] == 2:
                print('check you code, mapped_dimension is wrong')
                image_data_mapped = False
        else:
            image_data_mapped = False
        return image_data_mapped

    def anti_feature_map(self, state):
        # test code
        state_shape = list(state.shape)
        state_shape.pop(-1)
        pixels = numpy.arcsin(numpy.abs(
            state.reshape(-1, self.para['mapped_dimension'])[:, 0])).reshape(state_shape) / self.para['theta']
        return pixels

    # shuffle the image_data correspond to image_label
    def rand_sort_data(self, image_data, image_label):
        numpy.random.seed(self.para['rand_index_seed'])
        rand_index = numpy.random.permutation(image_data.shape[0])
        image_data_rand_sorted = image_data[rand_index, :]
        image_label_rand_sorted = image_label[rand_index]
        return image_data_rand_sorted, image_label_rand_sorted

    # train_test_split for data
    def split_data(self, data, test_size):
        length = len(data)
        image_data = []
        image_label = []
        for name,label in data.values:
            tmp_img = io.imread(name, as_gray = True)
            tmp_img = transform.resize(tmp_img, self.para['resize_size'])
            image_data.append(tmp_img.flatten())
            image_label.append(label)
        image_data = np.array(image_data)
        image_label = np.array(image_label)
        rand_image_data,rand_image_label = self.rand_sort_data(image_data, image_label)
        train_size = int(length * (1 - test_size))
        return rand_image_data[:train_size], rand_image_label[:train_size], rand_image_data[train_size:], rand_image_label[train_size:]

        # restore the image path and label
    def Restore_Img_Path(self,Dst_Path, csv_path):
        '''
        Dst_Path: the content of Image file
        '''
        img_path = []
        img_label = []
        label_path_name = os.listdir(Dst_Path)
        tmp_index = numpy.arange(len(label_path_name))
        # le = LabelEncoder()
        # le.fit(label_path_name)
        label_number = 0
        for p in label_path_name:
            label_path = os.path.join(Dst_Path, p)
            if not os.path.isdir(label_path):
                continue
            # print("the label path is : ",label_path)
            img_name = os.listdir(label_path)
            for img in img_name:
                cpl_img_path = os.path.join(label_path, img)
                img_path.append(cpl_img_path)
                img_label.append(label_number)
            label_number += 1
        # img_label = le.transform(img_label)
        data = {"path":img_path,"label":img_label}
        img_df = pd.DataFrame(data)
        img_df.to_csv(csv_path,index=False)

    def Restore_Yale_Img_Path(self,Dst_Path, csv_path):
        '''
        Dst_Path: the content of Image file
        '''
        img_path = []
        img_label = []
        label_path_name = os.listdir(Dst_Path)
        
        label_number=1
        for sub_dir in label_path_name:
            sub_dir_list = os.listdir(os.path.join(Dst_Path,sub_dir))
            delete_item=[]
            for sub_file_name in sub_dir_list:
                if not ('info' in sub_file_name or 'Ambient' in sub_file_name):
                    img_path.append(os.path.join(Dst_Path, sub_dir, sub_file_name))
                    img_label.append(label_number)

            label_number += 1
         
        data = {"path":img_path,"label":img_label}
        img_df = pd.DataFrame(data)
        img_df.to_csv(csv_path,index=False)


    def calculate_running_time(self, mode='end'):
        if mode == 'start':
            self.tmp['start_time_cpu'] = time.clock()
            self.tmp['start_time_wall'] = time.time()
        elif mode == 'end':
            self.tmp['end_time_cpu'] = time.clock()
            self.tmp['end_time_wall'] = time.time()
            self.update_info['cost_time_cpu'].append(self.tmp['end_time_cpu'] - self.tmp['start_time_cpu'])
            self.update_info['cost_time_wall'].append(self.tmp['end_time_wall'] - self.tmp['start_time_wall'])

    def print_running_time(self, print_type=('wall')):
        if ('cpu' in print_type) or ('cpu' == print_type):
            print('This loop consumes ' + str(self.tmp['end_time_cpu']
                                              - self.tmp['start_time_cpu']) + ' cpu seconds.')
        if ('wall' in print_type) or ('wall' == print_type):
            print('This loop consumes ' + str(self.tmp['end_time_wall']
                                              - self.tmp['start_time_wall']) + ' wall seconds.')

    def is_converge(self):
        if self.para['converge_type'] == 'cost function':
            loops_learned = self.update_info['loops_learned']
            cost_function_loops = self.update_info['cost_function_loops']
            self.update_info['is_converged'] = bool(
                ((cost_function_loops[loops_learned - 1] - cost_function_loops[loops_learned]) /
                 abs(cost_function_loops[loops_learned - 1])) < self.para['converge_accuracy'])
            if self.update_info['is_converged']:
                if self.update_info['step'] > self.para['step_accuracy']:
                    self.update_info['step'] /= self.para['step_decay_rate']
                    print('update step reduces to ' + str(self.update_info['step']))
                    self.update_info['is_converged'] = False

    def print_converge_info(self):
        print(self.para['converge_type'] + ' is converged at ' + str(self.update_info['cost_function'])
              + '. Program terminates')
        print('Train ' + str(self.update_info['loops_learned']) + ' loops')

    def generate_update_info(self):
        self.update_info['update_position'] = 'unknown'
        self.update_info['update_direction'] = +1
        self.update_info['loops_learned'] = 0
        self.update_info['cost_function_loops'] = list()
        self.update_info['cost_time_cpu'] = list()
        self.update_info['cost_time_wall'] = list()
        self.update_info['step'] = self.para['update_step']
        self.update_info['is_converged'] = 'untrained'
        self.update_info['update_mode'] = self.para['update_mode']

    # Get batch transform operations use it upon input images
    def deal_data(self, image_data, data_deal_method=tuple(), reverse_mode='off'):
        tmp_image_data = image_data.copy()
        if numpy.ndim(tmp_image_data) == 1:
            tmp_image_data.shape = (1,) + tmp_image_data.shape
        if len(data_deal_method) == 0:
            data_deal_method = self.para['data_deal_method']
        if reverse_mode == 'off':
            for method in data_deal_method:
                tmp_image_data = self.deal_data_once(tmp_image_data.copy(), method, reverse_mode=reverse_mode)
        elif reverse_mode == 'on':
            for method in reversed(data_deal_method):
                tmp_image_data = self.deal_data_once(tmp_image_data.copy(), method, reverse_mode=reverse_mode)
        return tmp_image_data

    # Take some operations such as rgb2gray/resize/normalization/noise/dct and so on to transform the image, 
    # Just worked as a preprocess procedure.
    def deal_data_once(self, image_data, method, reverse_mode='off'):
        tmp_image_data = image_data.copy()
        if ('rgb2gray' == method) & (reverse_mode == 'off'):
            tmp = list()
            for jj in range(tmp_image_data.shape[0]):
                tmp.append(
                    cv2.cvtColor(tmp_image_data[jj].reshape(
                        self.data_info['origin_shape']), cv2.COLOR_RGB2GRAY).flatten())
            tmp_image_data = numpy.array(tmp)
            self.data_info['shape'] = (self.data_info['origin_shape'][0], self.data_info['origin_shape'][1])
        if ('resize' == method) & (reverse_mode == 'off'):
            tmp = list()
            for jj in range(tmp_image_data.shape[0]):
                tmp.append(cv2.resize(
                    tmp_image_data[jj].reshape(self.data_info['origin_shape']),
                    self.para['resize_size']).flatten())
            tmp_image_data = numpy.array(tmp)
            self.data_info['shape'] = (
                self.para['resize_size'][1], self.para['resize_size'][0])
            self.data_info['n_feature'] = tmp_image_data.shape[1]
        if ('normalization' == method) & (reverse_mode == 'off'):
            tmp_image_data = tmp_image_data / tmp_image_data.max()
        if ('noise' == method) & (reverse_mode == 'off'):
            tmp_image_data = skimage.util.random_noise(
                tmp_image_data, mean=0,
                var=self.para['var_noise'],
                seed=self.para['noise_seed'])
        if ('dct' == method) & (reverse_mode == 'off'):
            for jj in range(tmp_image_data.shape[0]):
                tmp_image_data[jj] = cv2.dct(
                    tmp_image_data[jj].reshape(self.data_info['shape'])).flatten()
        if ('dct' == method) & (reverse_mode == 'on'):
            for jj in range(tmp_image_data.shape[0]):
                tmp_image_data[jj] = cv2.idct(
                    tmp_image_data[jj].reshape(self.data_info['shape'])).flatten()
        if ('standardize' == method) & (reverse_mode == 'off'):
            self.data_info['standardize'] = (tmp_image_data.max() - tmp_image_data.min(), tmp_image_data.min())
            tmp_image_data = tmp_image_data - self.data_info['standardize'][1]
            tmp_image_data = tmp_image_data / self.data_info['standardize'][0]
        if ('standardize' == method) & (reverse_mode == 'on'):
            tmp_image_data = tmp_image_data * self.data_info['standardize'][0]
            tmp_image_data = tmp_image_data + self.data_info['standardize'][1]
        if ('snake_like' == method) & (reverse_mode == 'off'):
            # testing code
            side_length = self.data_info['shape'][0]
            index_matrix = torch.zeros(side_length, side_length)
            index = 0
            for tt in range(2 * side_length - 2):
                for jj in range(side_length):
                    if -1 < tt - jj < side_length:
                        index_matrix[jj, tt - jj] = index
                        index += 1
            index = index_matrix.reshape(-1)
            tmp_image_data = tmp_image_data[:, index.int()]
        if ('snake_like' == method) & (reverse_mode == 'off'):
            # testing code
            side_length = self.data_info['shape'][0]
            index_matrix = torch.zeros(side_length, side_length)
            index = 0
            for tt in range(2 * side_length - 2):
                for jj in range(side_length):
                    if -1 < tt - jj < side_length:
                        index_matrix[jj, tt - jj] = index
                        index += 1
            index = index_matrix.reshape(-1)
            tmp_image_data[:, index.int()] = tmp_image_data
        if ('split_test' == method) & (reverse_mode == 'off'):
            tmp_shapes = self.para['split_shapes']
            tmp_shapeb = self.para['split_shapeb']
            shapea = (tmp_shapes[0] * tmp_shapeb[0], tmp_shapes[1] * tmp_shapeb[1])
            index_xx = torch.zeros(tmp_shapeb + tmp_shapes)
            for xx in range(tmp_shapes[0]):
                for yy in range(tmp_shapes[1]):
                    index_xx[0, 0, xx, yy] = xx * shapea[1] + yy
            for xx in range(tmp_shapeb[0]):
                for yy in range(tmp_shapeb[1]):
                    index_xx[xx, yy, :, :] = index_xx[0, 0, :, :] + yy * tmp_shapes[1] \
                                             + xx * shapea[1] * tmp_shapes[0]
            tmp_image_data = image_data[:, index_xx.int().reshape(-1)]
            tmp_image_data = tmp_image_data.reshape(
                image_data.shape[0] * numpy.prod(tmp_shapeb), numpy.prod(tmp_shapes))
            self.data_info['shape'] = tmp_shapes
        if ('split_test' == method) & (reverse_mode == 'on'):
            tmp_shapes = self.para['split_shapes']
            tmp_shapeb = self.para['split_shapeb']
            shapea = (tmp_shapes[0] * tmp_shapeb[0], tmp_shapes[1] * tmp_shapeb[1])
            index_xx = torch.zeros(tmp_shapeb + tmp_shapes)
            for xx in range(tmp_shapes[0]):
                for yy in range(tmp_shapes[1]):
                    index_xx[0, 0, xx, yy] = xx * shapea[1] + yy
            for xx in range(tmp_shapeb[0]):
                for yy in range(tmp_shapeb[1]):
                    index_xx[xx, yy, :, :] = index_xx[0, 0, :, :] + yy * tmp_shapes[1] \
                                             + xx * shapea[1] * tmp_shapes[0]
            tmp_image_data = tmp_image_data.reshape(-1, numpy.prod(tmp_shapeb + tmp_shapes))
            tmp_image_data[:, index_xx.int().reshape(-1)] = tmp_image_data.copy()
        if ('test' == method) & (reverse_mode == 'off'):
            tmp_image_data = tmp_image_data + 5
            tmp_image_data = tmp_image_data / 25
        if ('test' == method) & (reverse_mode == 'on'):
            tmp_image_data = tmp_image_data * 25
            tmp_image_data = tmp_image_data - 5
        return tmp_image_data


