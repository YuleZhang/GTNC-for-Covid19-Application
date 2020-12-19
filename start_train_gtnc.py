# -*- encoding: utf-8 -*-
from library import MPSMLclass
from library import Parameters
import matplotlib.pyplot as plt
import numpy as np
import json
import copy

para = Parameters.gtnc()
# get the relation between cutting dimention and testing accuracy
batch_bond = []
batch_acc = []
saved_para = dict()
for i in range(1, 9):
    bond = 2 ** i # get the cutting bond
    para['virtual_bond_limitation'] = bond
    print('The current cutting bond is %d' % bond)
    A = MPSMLclass.GTNC(para=para, device='cpu')  # change device='cuda' to use GPU
    A.training_gtn()  # if the GTN are not trained
    acc = A.calculate_accuracy('test')
    print('Bond {0} acc is {1}'.format(bond, acc))
    batch_bond.append(bond)
    # para[''] = acc
    batch_acc.append(acc)
    counterpart_para = copy.deepcopy(para)
    for item in counterpart_para.keys():
        counterpart_para[item] = str(counterpart_para[item])
    counterpart_para['Accuracy'] = acc
    saved_para["bond"+str(bond)] = counterpart_para

f2 = open('new_json.json', 'a+')
json.dump(saved_para, f2)
f2.close()
bond_index = np.arange(len(batch_bond))
plt.plot(bond_index, batch_acc,color='k',linestyle='-',marker = 'o',markerfacecolor='r',markersize = 10)
plt.plot(bond_index, [0.95 for i in bond_index])
plt.xticks(bond_index,batch_bond)
plt.xlabel('χ(GTNC)')
plt.ylabel('test accuracy')
plt.savefig('test.jpg')
plt.show()

