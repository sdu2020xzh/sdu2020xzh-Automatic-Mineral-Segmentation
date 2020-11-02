import argparse
import Models, LoadBatches
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras import backend as K
from keras.losses import categorical_crossentropy
from lovasz_loss_keras import lovasz_softmax_flat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path",
                    type=str, default='weights/rock_model/resnet50unet/resnet50Unet_rock.h5')  # 保存模型
parser.add_argument("--train_images", type=str, default="data/rock/images/")  # 训练集路径（直接在对应路径放图片即可）
parser.add_argument("--train_annotations", type=str, default="data/rock/segmentation/")  # 训练集标注图像路径
parser.add_argument("--n_classes", type=int, default=5)  # 矿物种类
parser.add_argument("--input_height", type=int, default=512)
parser.add_argument("--input_width", type=int, default=512)  # 输入图片尺寸

parser.add_argument('--validate', action='store_true')
parser.add_argument("--val_images", type=str, default="data/rock/val/")  # 验证集路径
parser.add_argument("--val_annotations", type=str, default="data/rock/val_seg/")  # 验证集标注路径

parser.add_argument("--epochs", type=int, default=40)  # 训练轮次
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--val_batch_size", type=int, default=2)  # 每批图像数量
parser.add_argument("--load_weights", type=str, default="")

parser.add_argument("--model_name", type=str, default="resnet50Unet")
# parser.add_argument("--optimizer_name", type=str, default="adadelta")
parser.add_argument("--optimizer_name", type=str, default="Adam")

# 初始化
args = parser.parse_args()
train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
optimizer_name = args.optimizer_name
model_name = args.model_name
val_images_path = args.val_images
val_segs_path = args.val_annotations
val_batch_size = args.val_batch_size


# 损失函数和准确率
def generalized_dice(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(-1, 4))
    y_pred = K.reshape(y_pred, shape=(-1, 4))
    sum_p = K.sum(y_pred, -2)
    sum_r = K.sum(y_true, -2)
    sum_pr = K.sum(y_true * y_pred, -2)
    weights = K.pow(K.square(sum_r) + K.epsilon(), -1)
    generalized_dice = (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))
    return generalized_dice


def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice(y_true, y_pred)


def custom_loss(y_true, y_pred):
    return generalized_dice_loss(y_true, y_pred) + 1.25 * categorical_crossentropy(y_true, y_pred)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


def iou(y_true, y_pred, label:  int):
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    num_labels = K.int_shape(y_pred)[-1] - 1
    mean_iou = K.variable(0)
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)
    return mean_iou / num_labels


def mean_IoU_loss(y_true, y_pred):
    return 1 - mean_iou(y_true, y_pred)


def custom_IoU_CCE(y_true, y_pred):
    return 0.5 * categorical_crossentropy(y_true, y_pred) + 1.5 * mean_IoU_loss(y_true, y_pred)


def Lovasz_loss_function(y_true, y_pred):
    lovasz = lovasz_softmax_flat(y_pred, y_true, classes="present")
    return lovasz


# 创建模型
def check_print(filepath):
    modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet,
                'vgg_unet': Models.VGGUnet.VGGUnet,
                'resnet50Unet': Models.resnet_unet.resnet_50_unet,
                'resnet101Unet': Models.resnet_unet.resnet_101_unet}
    modelFN = modelFns[model_name]
    m = modelFN(n_classes, input_height=input_height, input_width=input_width)
    if os.path.exists(filepath):
        m.load_weights(filepath)
    m.summary()
    m.compile(loss=[custom_IoU_CCE],
              optimizer=optimizer_name,
              metrics=[mean_iou])
    print('Model Compiled')
    return m


# 训练
if __name__ == '__main__':
    filepath = 'weights/rock_model/wjy_resnet50unet/yh/resnet50Unet_wjy.h5'
    model = check_print(filepath)
    print("Model output shape", model.output_shape)
    output_height = model.outputHeight
    output_width = model.outputWidth
G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width)
G2 = LoadBatches.imageSegmentationGenerator
(val_images_path, val_segs_path, val_batch_size, n_classes, input_height, input_width, output_height, output_width)
checkpoint = ModelCheckpoint(save_weights_path.split(".")[0] + "_" + "{epoch:02d}" + "_" + "{val_mean_iou:.2f}" + ".h5",
                             monitor='val_mean_iou', verbose=1, save_best_only=False)
# 记录训练数据
history = model.fit_generator(G, validation_data=G2, epochs=epochs, steps_per_epoch=1024, validation_steps=1024,
                              callbacks=[checkpoint])
print('Saved trained model at %s ' % filepath)
print(history.history.keys())
plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])
plt.title('model dice value')
plt.ylabel('mean_iou')
plt.xlabel('epoch')
plt.savefig(filepath.split(".")[0] + "_IoU1" + ".png", dpi=300)
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(filepath.split(".")[0] + "_loss1" + ".png", dpi=300)
plt.show()




