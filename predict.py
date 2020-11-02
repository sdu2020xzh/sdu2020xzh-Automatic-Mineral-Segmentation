#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import Models, LoadBatches
from PIL import Image
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str, default='weights/rock_model/resnet50unet/resnet50Unet_rock_75_0.73.h5')  #调用模型路径
parser.add_argument("--epoch_number", type=int, default=9)
parser.add_argument("--test_images", type=str, default="data/rock/val/")  #测试集图像路径
parser.add_argument("--output_path", type=str, default="data/predictions/")  #预测结果保存路径
parser.add_argument("--input_height", type=int, default=512)
parser.add_argument("--input_width", type=int, default=512)  #输入图片尺寸
parser.add_argument("--model_name", type=str, default="resnet50Unet")  #调用模型
parser.add_argument("--n_classes", type=int, default=5)  #矿物种类

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width = args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
                'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32,
                'mobile-segnet': Models.mobile_segnet.mobilenet_segnet,
                'resnet50Unet':Models.resnet_unet.resnet_50_unet}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.load_weights(args.save_weights_path)
m.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth
testroot = images_path
images = os.listdir(testroot)
print(images)

colors = np.array([[0, 0, 0],
                   [128, 0, 0],
                   [0, 128, 0],
                   [0, 0, 128],
                   [128, 0, 128]
                   ], dtype='uint8')

for imgName in images:
    img = cv2.imread(os.path.join(testroot, imgName), 1)
    X = LoadBatches.getImageArr(img, args.input_width, args.input_height)
    pr = m.predict(np.array([X]))[0]

    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    origin_img = Image.open(os.path.join(testroot, imgName))
    width, height = origin_img.size
    seg_img = Image.fromarray(np.uint8(seg_img))
    seg_img = seg_img.resize((width, height))
    origin_img = origin_img.convert('RGBA')
    seg_img = seg_img.convert('RGBA')
    pred_image = Image.blend(origin_img, seg_img, 0.5)
    pred_image.save("data/predictions/" + imgName.split(".")[0] + ".png")