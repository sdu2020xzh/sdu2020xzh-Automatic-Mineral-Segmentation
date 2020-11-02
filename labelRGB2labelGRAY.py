import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

corlor = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [128, 128, 0],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128]
                                 ], dtype='uint8')
print(corlor.shape)
#print(corlor[3][1])
tpath = "data/voc_train_segmentation"
vpath = "data/voc_val_segmentation"
ts = "data/train_segmentation"
vs = "data/val_segmentation"
train_labels = os.listdir(tpath)
val_labels = os.listdir(vpath)
count1 = 0
count2 = 0
for label in train_labels:
    labelpath = os.path.join(tpath, label)
    annotation = np.array(Image.open(labelpath))  # 根据自己的实际路径更改路径名
    temp = annotation[:,:,0]
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            for k in range(corlor.shape[0]):
                if (annotation[i][j] == corlor[k]).all():
                    temp[i][j] = k
    labelgray = Image.fromarray(temp)
    labelgray.save(os.path.join(ts, label))
    count1 +=1
    print("已处理%s张训练标签"%count1)
for label in val_labels:
    labelpath = os.path.join(vpath, label)
    annotation = np.array(Image.open(labelpath))  # 根据自己的实际路径更改路径名
    temp = annotation[:,:,0]
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            for k in range(0,corlor.shape[0]):
                if (annotation[i][j] == corlor[k]).all():
                    temp[i][j] = k
    labelgray = Image.fromarray(temp)
    labelgray.save(os.path.join(vs, label))
    count2 +=1
    print("已处理%s张验证集标签"%count2)

