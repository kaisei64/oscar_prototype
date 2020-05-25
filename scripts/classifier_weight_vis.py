import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from util_func import parameter_use
import pandas as pd
import matplotlib.pyplot as plt

prototype = "15"
train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_1500.pkl')
# train_net = parameter_use(f'./result/pkl/prototype_{prototype}/train_model_epoch1500_{prototype}.pkl')
classifier_weight = train_net.classifier[0].weight.cpu().detach().numpy().T

class_name = [i for i in range(10)]
df = pd.DataFrame(classifier_weight, columns=class_name)
# print(df)
fig, ax = plt.subplots(figsize=((len(df.columns))*1.2, (len(df))*0.4))
ax.axis('off')
colors = [["silver" if classifier_weight[j][i] - min(classifier_weight[j]) < abs(min(classifier_weight[j])) * 0.4
           else "w" for i in range(len(classifier_weight[0]))]
          for j in range(len(classifier_weight[:, 0]))]
tbl = ax.table(cellText=df.values.round(3),
               bbox=[0.05, 0, 0.9, 1],
               colLabels=df.columns,
               rowLabels=df.index,
               cellColours=colors)
tbl.set_fontsize(30)
plt.savefig(f'./result/png/prototype_{prototype}/classifier_weight.png')
# plt.show()