# 导入所需库及网络模型
import itertools
import pandas as pd
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.python.ops.confusion_matrix import confusion_matrix
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import *
from tensorflow.keras.applications import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1),
              name=None):  # 函数定义 这个函数接受输入张量 x，卷积核滤波器数目 nb_filter，卷积核大小 kernel_size，填充方式 padding，步幅 strides 和命名参数 name。
    # 名称处理：这段代码根据传入的 name 参数来生成用于批归一化层和卷积层的名称
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


# 调用TF官方模型以及ImageNet权重进行迁移学习
def merge_model():
    from densenet169 import DenseNet169 as model1
    from densenet169 import DenseNet169 as model2
    from VGG16 import VGG16 as model3
    inpt1 = Input(shape=(224, 224, 3))
    inpt2 = Input(shape=(224, 224, 3))
    inpt3 = Input(shape=(224, 224, 3))
    model_1 = model1(input_tensor=inpt1)
    model_2 = model2(input_tensor=inpt1)
    model_3 = model3(input_tensor=inpt1)
    r1 =model_1.output
    r2 = model_2.output
    r3 = model_3.output
    a1 = concatenate([r1, r2, r3], axis=3)
    a1 = Conv2d_BN(a1, 3, (7, 7), strides=(1, 1), padding='same', name='0701')
    x = GlobalAveragePooling2D()(a1)
    x = BatchNormalization()(x)
    x =tf.nn.gelu(x)
    a6 = Dense(1024)(x)
    a7 = tf.nn.gelu(a6)
    a8 = Dropout(0.2)(a7)
    a9 = Dense(676)(a8)
    a10 = Dropout(0.2)(a9)
    x = Dense(4, activation="softmax")(a10)
    model = Model(inputs=[inpt1, inpt2, inpt3], outputs=x)
    return model


def main():
    im_height = 224  # 图像长度
    im_width = 224   # 图像宽度
    num_classes = 4  # 类别数

    labels = ['class_0', 'class_1', 'class_2', 'class_3']

    # 设置三个数据集的路径
    data_root1 = "E:/2022grade/fxl/New model/公共数据集/ACRIMA/ACRIMA/PARTITIONED/val_dataset1"
    data_root2 = "E:/2022grade/fxl/New model/公共数据集/ACRIMA/ACRIMA/PARTITIONED/val_dataset2"
    data_root3 = "E:/2022grade/fxl/New model/公共数据集/ACRIMA/ACRIMA/PARTITIONED/val_dataset3"

    # 初始化ImageDataGenerator并加载三个数据集
    test_image_generator = ImageDataGenerator(rescale=1. / 255)
    test_data_gen1 = test_image_generator.flow_from_directory(data_root1, batch_size=32, shuffle=False,
                                                               target_size=(im_height, im_width), class_mode='categorical')
    test_data_gen2 = test_image_generator.flow_from_directory(data_root2, batch_size=32, shuffle=False,
                                                               target_size=(im_height, im_width), class_mode='categorical')
    test_data_gen3 = test_image_generator.flow_from_directory(data_root3, batch_size=32, shuffle=False,
                                                               target_size=(im_height, im_width), class_mode='categorical')

    # 导入网络模型并加载权重
    model = merge_model()
    weights_path = "E:/2022grade/fxl/New model/weights/1ACRIMA.h5"
    assert os.path.exists(weights_path), "Weights file does not exist."
    model.load_weights(weights_path)
    # 启用tf.function的即刻执行，而不是作为跟踪图函数运行，防止graph报错
    tf.config.experimental_run_functions_eagerly(True)
    # 预测三个数据集的结果
    y_pred1 = model.predict(test_data_gen1)
    y_pred2 = model.predict(test_data_gen2)
    y_pred3 = model.predict(test_data_gen3)

    # 合并预测结果
    y_pred = np.concatenate([y_pred1, y_pred2, y_pred3],  axis=1)
    predictions = np.array(list(map(lambda x: np.argmax(x), y_pred)))

    # 获取每个数据集的真实标签
    y_true1 = test_data_gen1.classes
    y_true2 = test_data_gen2.classes
    y_true3 = test_data_gen3.classes
    y_true = np.concatenate([y_true1, y_true2, y_true3],  axis=1)
    y_label = np.float32(preprocessing.label_binarize(y_true, classes=[0, 1, 2, 3, 4]))  # 将标签二值化并转为float32类型以供AUC计算使用

    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_label[:, i], y_pred[:, i])  # 计算第i个类别的FPR，TPR和阈值
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])  # 计算第i个类别的AUC
    ###Plot all ROC curves
    plt.figure(figsize=(7, 7))

    colors = itertools.cycle(['red', 'blue', 'green', '#800080'])
    classnames = itertools.cycle(['Normal', 'Early', 'Intermediate', 'Terminal'])
    for i, color, name in zip(range(num_classes), colors, classnames):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, markevery=20,
                 label='{0} (AUC = {1:0.4f})'
                       ''.format(name, roc_auc[i]))

    plt.plot([0, 1], [0, 1], '--', color='#747d8c')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    plt.title('ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.show()

   # 打印评价指标
    print('Testing:')
    print('Accuracy score is :', metrics.accuracy_score(y_true, predictions))  # 计算准确率
    print('Precision score is :', metrics.precision_score(y_true, predictions, average='macro'))  # 计算精确率
    print('Recall score is :', metrics.recall_score(y_true, predictions, average='macro'))  # 计算召回率
    print('F1 Score is :', metrics.f1_score(y_true, predictions, average='macro'))  # 计算F1分数
    print('Cohen Kappa Score:', metrics.cohen_kappa_score(y_true, predictions))  # 计算Kappa系数

    def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=(6.4, 4.8)):
        """
        A function to create a colored and labeled confusion matrix matplotlib figure
        given true labels and preds.
        Args:
            cmtx (ndarray): confusion matrix.
            num_classes (int): total number of classes.
            class_names (Optional[list of strs]): a list of class names.
            figsize (Optional[float, float]): the figure size of the confusion matrix.
                If None, default to [6.4, 4.8].

        Returns:
            img (figure): matplotlib figure.
        """
        if class_names is None or type(class_names) != list:
            class_names = [str(i) for i in range(num_classes)]

        figure = plt.figure(figsize=figsize)
        plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cmtx.max() / 2.0
        for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
            color = "white" if cmtx[i, j] > threshold else "black"
            plt.text(
                j,
                i,
                format(cmtx[i, j]) if cmtx[i, j] != 0 else "0",
                horizontalalignment="center",
                color=color,
            )

        # plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        # plt.savefig('plt/' + "qgy_fo4_matrix.png")
        return figure

    confusion = confusion_matrix(predictions, y_pred)  # 4class
    plot_confusion_matrix(confusion, num_classes=4)
    plt.show()
    print(confusion)
    muticlass = multilabel_confusion_matrix(predictions, y_pred)
    print(muticlass)

    for i in range(len(muticlass)):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        TP += muticlass[i][0, 0]
        TN += muticlass[i][1, 1]
        FN += muticlass[i][0, 1]
        FP += muticlass[i][1, 0]
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        spe = TN / (TN + FP)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('*****************The {} class********************'.format(i))
        print("{}th Precision: ".format(i) + str(p))
        print("{}th Sensitivity:".format(i) + str(r))
        print("{}th Specificity:".format(i) + str(spe))
        print("{}th F1 score (F-measure): ".format(i) + str(F1))
        print("{}th Accuracy: ".format(i) + str(acc))



if __name__ == '__main__':
    main()
