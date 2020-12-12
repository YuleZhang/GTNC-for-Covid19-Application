# -*- encoding: utf-8 -*-
'''
2020-11-29
Some basic operations of tensors are encapsulated, which can be transplanted, and some comments are added.
'''
import numpy
import torch

# 平行外积
def outer_parallel(a, *matrix):
    # need optimization
    for b in matrix:
        a = (a.repeat(b.shape[1], 1).reshape(a.shape + (-1,))
             * b.repeat(a.shape[1], 0).reshape(a.shape + (-1,))).reshape(a.shape[0], -1)
    return a

# 张量外积，阶数不发生变化
def outer(a, *matrix):
    for b in matrix:
        a = numpy.outer(a, b).flatten()
    return a

# 张量a和张量b收缩，仅收缩各自一条腿
def tensor_contract(a, b, index):
    '''
    parameter:
        a: numpy.array,Tensor a to be contracted
        b: numpy.array,Tensor b to be contracted
        index: Two-dimensional list,indicated the contracted index in a/b,such as [[1], [0]]
    return:
        numpy.array,the contracted result(matrix) of tensor a and b.
    '''
    ndim_a = numpy.array(a.shape)
    ndim_b = numpy.array(b.shape)
    order_a = numpy.arange(len(ndim_a)) # 生成各个阶对应的索引
    order_b = numpy.arange(len(ndim_b))
    order_a_contract = numpy.array(order_a[index[0]]).flatten()
    order_b_contract = numpy.array(order_b[index[1]]).flatten()
    order_a_hold = numpy.setdiff1d(order_a, order_a_contract) # 剔除order_a_contract在order_a中的元素，并进行unique
    order_b_hold = numpy.setdiff1d(order_b, order_b_contract) #　剔除order_b_contract在order_b中的元素，并进行unique
    hold_shape_a = ndim_a[order_a_hold].flatten()
    hold_shape_b = ndim_b[order_b_hold].flatten()
    # hold_shape_a.prod返回数组元素乘积，也就是将待合并的张量reshape成一个(x,y)的矩阵，其中x是剩余阶数的维度积，y是待收缩张量的维数
    return numpy.dot(
        a.transpose(numpy.concatenate([order_a_hold, order_a_contract])).reshape(hold_shape_a.prod(), -1), 
        b.transpose(numpy.concatenate([order_b_contract, order_b_hold])).reshape(-1, hold_shape_b.prod()))\
        .reshape(numpy.concatenate([hold_shape_a, hold_shape_b]))

# 张量svd分解
def tensor_svd(tmp_tensor, index_left='none', index_right='none'):
    '''
    parameter:
        tmp_tensor: numpy.array,default=none, Tensor to be decomposed
        index_left: int,default=none. range 0~2
        index_right: int,default=none.range 0~2
        Note that index_left and index_right cannot be none at the same time.
    return:
        decomposed result u,l,v
    '''
    tmp_shape = numpy.array(tmp_tensor.shape) 
    tmp_index = numpy.arange(len(tmp_tensor.shape))  # 很巧妙的生成各个阶的索引
    if index_left == 'none':
        index_right = tmp_index[index_right].flatten()
        index_left = numpy.setdiff1d(tmp_index, index_right) # 剔除全指标中index_right这个指标
    elif index_right == 'none':
        index_left = tmp_index[index_left].flatten()
        index_right = numpy.setdiff1d(tmp_index, index_left)
    index_right = numpy.array(index_right).flatten()
    index_left = numpy.array(index_left).flatten()
    tmp_tensor = tmp_tensor.permute(tuple(numpy.concatenate([index_left, index_right])))
    tmp_tensor = tmp_tensor.reshape(tmp_shape[index_left].prod(), tmp_shape[index_right].prod())# reshape成矩阵之后进行分解
    u, l, v = torch.svd(tmp_tensor) # svd分解并返回结果
    return u, l, v