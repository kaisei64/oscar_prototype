import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

prototype = "2"
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch500_{prototype}.pkl')
classifier_weight = train_net.classifier[0].weight.cpu().detach().numpy().T

class_name = [i for i in range(10)]
prototype_name = [chr(ord('A') + i) for i in range(int(prototype))]
df = pd.DataFrame(classifier_weight, columns=class_name)
# print(df)
fig, ax = plt.subplots(figsize=((len(df.columns))*1.2, (len(df))*0.4))
ax.axis('off')
# Color the weights that contribute a lot
# colors = [["silver" if classifier_weight[j][i] - min(classifier_weight[j]) < abs(min(classifier_weight[j])) * 0.4
#            else "w" for i in range(len(classifier_weight[0]))]
#           for j in range(len(classifier_weight[:, 0]))]
vals = np.around(df.values, 2)
normal = plt.Normalize(vals.min()-1, vals.max()+1)
tbl = ax.table(cellText=df.values.round(3),
               bbox=[0.05, 0, 0.9, 1],
               colLabels=df.columns,
               rowLabels=prototype_name,
               # cellColours=colors)
               cellColours=plt.cm.gray(normal(vals)))
tbl.set_fontsize(30)
plt.savefig(f'./result/png/prototype_{prototype}/classifier_weight.png')
# plt.show()
