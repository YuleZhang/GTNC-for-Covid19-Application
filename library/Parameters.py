import numpy
import torch
import os


def gtn(para=dict()):
    para.update(ml())
    para.update(program())
    para.update(mps())
    para.update(feature_map())
    para.update(training())
    # Machine Learning Parameter
    para['classifier_type'] = 'GTN'
    para['dataset'] = 'mnist'
    para['data_deal_method'] = ['normalization']
    para['resize_size'] = (14, 14)
    # para['resize_size'] = (14, 14)
    # Program Parameter
    # MPS Parameter
    para['physical_bond'] = 2
    para['virtual_bond_limitation'] = 8
    para['mps_cutoff'] = -1
    para['mps_rand_seed'] = 1
    # Feature Map Parameter
    para['theta'] = (numpy.pi/2) 
    para['mapped_dimension'] = 2
    # Training Parameter
    # para['training_label'] = [0]  # para['training_label'] should be a list
    para['training_label'] = list(range(10))  # para['training_label'] should be a list
    para['n_training'] = 1000  # an int or 'all'
    para['update_mode'] = 'one_dot'  # one_dot or two_dots
    return para


def gtnc(para=dict()):
    para.update(ml())
    para.update(program())
    para.update(mps())
    para.update(feature_map())
    para.update(training())
    # Machine Learning Parameter
    para['classifier_type'] = 'GTNC'
    para['dataset_content'] = 'dataset'
    para['dataset'] = 'X-Ray Image DataSet'
    # Program Parameter
    para['dtype'] = torch.float64
    para['data_deal_method'] = ['normalization']
    para['resize_size'] = (32, 32)
    # MPS Parameter
    para['physical_bond'] = 2
    para['virtual_bond_limitation'] = 32
    para['mps_cutoff'] = -1
    para['mps_rand_seed'] = 1
    # Feature Map Parameter
    para['theta'] = (numpy.pi/2)
    para['mapped_dimension'] = 2
    # para['map_module'] = 'linear_map'
    # para['map_module'] = 'sqrt_linear_map'
    para['map_module'] = 'linear_map'
    # Training Parameter
    para['training_label'] = [[0],[1],[2]]  # para['training_label'] should be a list
    # para['training_label'] = [[3], [8]]
    para['n_training'] = 'all'  # an int or 'all'
    para['update_mode'] = 'one_dot'  # one_dot or two_dots ()
    return para


def ml(para=dict()):
    base_path = os.path.dirname(__file__) + os.path.sep + ".."
    para['dataset'] = 'mnist'
    para['path_dataset'] = os.path.join(base_path,  "dataset/img_dict_cls3.csv") # "../dataset/img_dict_cls3.csv"
    para['data_type'] = ['train', 'test']
    para['classifier_type'] = 'None'
    para['sort_module'] = 'rand' # define the way to sort the dataset,rand mean shuffle the training data
    para['divide_module'] = 'label'
    para['save_data_path'] = os.path.join(base_path, 'data_trained/')
    para['rand_index_mode'] = 'on'
    para['rand_index_seed'] = 1
    para['data_deal_method'] = ['normalization']
    para['resize_size'] = (5, 5)
    para['split_shapes'] = (8, 12)
    para['split_shapeb'] = (20, 20)
    return para


def program(para=dict()):
    para['rough_mode'] = 'off'
    para['dtype'] = torch.float64
    para['device'] = 'cpu'
    return para


def mps(para=dict()):
    para['physical_bond'] = 2
    para['virtual_bond_limitation'] = 8
    para['tensor_network_type'] = 'MPS'
    para['mps_cutoff'] = 1e-9
    para['mps_normalization_mode'] = 'on'
    para['move_label_mode'] = 'off'
    para['tensor_initialize_type'] = 'ones'
    para['tensor_initialize_bond'] = 'max'
    para['mps_rand_seed'] = 1
    return para


def feature_map(para=dict()):
    para['map_module'] = 'many_body_Hilbert_space'
    para['theta'] = (numpy.pi/2) 
    para['mapped_dimension'] = 2
    return para


def training(para=dict()):
    para['training_label'] = [[3], [8]]
    para['n_training'] = 40  # an int or 'all'
    para['update_step'] = 2e-1
    para['step_decay_rate'] = 5
    para['step_accuracy'] = 5e-3
    para['normalization_update_mode'] = 'on'
    para['update_mode'] = 'one_dot'  # one_dot or two_dots
    para['update_method'] = 'SZZ'
    para['converge_type'] = 'cost function'
    para['converge_accuracy'] = 1e-2
    para['rand_index_mode'] = 'on'
    para['rand_index_seed'] = 1
    para['tensor_acc'] = 1e-8
    return para


