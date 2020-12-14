# -*- encoding: utf-8 -*-
from library import MPSMLclass
from library import Parameters
import matplotlib.pyplot as plt
import numpy as np

para = Parameters.gtnc()
# get the relation between cutting dimention and testing accuracy
batch_bond = []
batch_acc = []
for i in range(1, 8):
    bond = 2 ** i # get the cutting bond
    para['virtual_bond_limitation'] = bond
    print('The current cutting bond is %d' % bond)
    A = MPSMLclass.GTNC(para=para, device='cpu')  # change device='cuda' to use GPU
    A.training_gtn()  # if the GTN are not trained
    acc = A.calculate_accuracy('test')
    print('Bond {0} acc is {1}'.format(bond, acc))
    batch_bond.append(bond)
    batch_acc.append(acc)
bond_index = np.arange(len(batch_bond))
plt.plot(bond_index, batch_acc,color='k',linestyle='-',marker = 'o',markerfacecolor='r',markersize = 10)
plt.plot(bond_index, [0.95 for i in bond_index])
plt.xticks(bond_index,batch_bond)
plt.xlabel('χ(GTNC)')
plt.ylabel('test accuracy')
plt.show()

