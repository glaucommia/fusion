# 导入所需库以及网络模型
import tensorflow as tf
# 使用GPU加速计算
from opt_einsum.backends import tensorflow
from pandas import np
from tensorflow.python.keras.callbacks import ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # GPU自动分配显存
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import json
import os
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf

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

# def merge_model():
#     name = 'merge'
#     inpt1 = Input(shape=(224, 224, 3))
#     inpt2 = Input(shape=(224, 224, 3))
#     inpt3 = Input(shape=(224, 224, 3))
#     o1 = Conv2d_BN(inpt1, 3, (7, 7), strides=(1, 1), padding='same')
#     o2 = Conv2d_BN(inpt2, 3, (7, 7), strides=(1, 1), padding='same')
#     o3 = Conv2d_BN(inpt3, 3, (7, 7), strides=(1, 1), padding='same')
#
#     a1 = concatenate([o1, o2, o3], axis=3)
#     a1 = Conv2d_BN(a1, 3, (7, 7), strides=(1, 1), padding='same')
#     baseModel1 = ResNet101(weights="imagenet", include_top=False, )(a1)
#
#     x = GlobalAveragePooling2D()(baseModel1)
#     x = BatchNormalization()(x)
#     x = Dense(4, activation="softmax")(x)
#     a6 = Dense(1024)(x)
#     a7 = tensorflow.compat.v1.nn.relu6(a6)
#     a8 = Dropout(0)(a7)
#     a9 = Dense(676)(a8)
#     a10 = Dropout(0)(a9)
#     a11 = Dense(4, activation="softmax")(a10)
#     model = Model(inputs=[inpt1, inpt2, inpt3], outputs=a11)
#     return model

# def newModel():
#     from densenet169 import DenseNet169 as model1
#     from ResNet101 import ResNet101 as model2
#     from VGG16 import VGG16 as model3
#     inpt1 = Input(shape=(224, 224, 3))
#     inpt2 = Input(shape=(224, 224, 3))
#     inpt3 = Input(shape=(224, 224, 3))
#     model_1 = model1(input_tensor=inpt1)
#     model_2 = model2(input_tensor=inpt1)
#     model_3 = model3(input_tensor=inpt1)
#     r1 =model_1.output
#     r2 = model_2.output
#     r3 = model_3.output
#     a1 = concatenate([r1, r2, r3], axis=3)
#     a1 = Conv2d_BN(a1, 3, (7, 7), strides=(1, 1), padding='same', name='0701')
#     x = GlobalAveragePooling2D()(a1)
#     x = BatchNormalization()(x)
#     x =tf.nn.gelu(x)
#     a6 = Dense(1024)(x)
#     a7 = tf.nn.gelu(a6)
#     a8 = Dropout(0.2)(a7)
#     a9 = Dense(676)(a8)
#     a10 = Dropout(0.2)(a9)
#     x = Dense(4, activation="softmax")(a10)
#     model = Model(inputs=[inpt1, inpt2, inpt3], outputs=x)
#     return model

def create_model(im_height, im_width, num_classes):
    # 定义输入
    input1 = Input(shape=(im_height, im_width, 3))
    input2 = Input(shape=(im_height, im_width, 3))
    input3 = Input(shape=(im_width, im_height, 3))
    from densenet169 import DenseNet169
    from ResNet101 import ResNet101
    from VGG16 import VGG16
    # DenseNet169 特征提取
    base_model1 = DenseNet169(include_top=False, weights='imagenet', input_tensor=input1)
    features1 = base_model1.output

    # ResNet101 特征提取
    base_model2 = ResNet101(include_top=False, weights='imagenet', input_tensor=input2)
    features2 = base_model2.output

    # VGG16 特征提取
    base_model3 = VGG16(include_top=False, weights='imagenet', input_tensor=input3)
    features3 = base_model3.output

    # 特征融合
    merged_features = concatenate([GlobalAveragePooling2D()(features1),
                                   GlobalAveragePooling2D()(features2),
                                   GlobalAveragePooling2D()(features3)])

    # 分类器
    x = Dense(1024, activation='relu')(merged_features)
    x = Dense(512, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # 创建模型
    model = Model(inputs=[input1, input2, input3], outputs=output)

    return model


def main():
    # 定义图像大小、批次大小、迭代次数和类别数
    im_height = 224  # 图像长度
    im_width = 224  # 图像宽度
    batch_size = 32 # 批次大小
    epochs = 1  # 迭代次数
    num_class = 4  # 类别数，根据您的类别数修改

    # 设置三个数据集的路径
    data_root1 = "E:/2022grade/fxl/New model/data_set/fundus"
    data_root2 = "E:/2022grade/fxl/New model/data_set/fundus"
    data_root3 = "E:/2022grade/fxl/New model/data_set/fundus"

    # 加载训练数据并进行断言检查
    train_dir1 = os.path.join(data_root1, "train")
    train_dir2 = os.path.join(data_root2, "train")
    train_dir3 = os.path.join(data_root3, "train")

    assert os.path.exists(train_dir1), "cannot find {}".format(train_dir1)
    assert os.path.exists(train_dir2), "cannot find {}".format(train_dir2)
    assert os.path.exists(train_dir3), "cannot find {}".format(train_dir3)

    # 创建保存权重的目录
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    # 图像预处理
    train_image_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        horizontal_flip=True,
        vertical_flip=True
    )

    # 对三个数据集分别进行flow_from_directory
    train_data_gen1 = train_image_generator.flow_from_directory(
        directory=train_dir1,
        batch_size=batch_size,
        shuffle=True,
        target_size=(im_height, im_width),
        class_mode='categorical'
    )
    total_train1 = train_data_gen1.n

    train_data_gen2 = train_image_generator.flow_from_directory(
        directory=train_dir2,
        batch_size=batch_size,
        shuffle=True,
        target_size=(im_height, im_width),
        class_mode='categorical'
    )
    total_train2 = train_data_gen2.n

    train_data_gen3 = train_image_generator.flow_from_directory(
        directory=train_dir3,
        batch_size=batch_size,
        shuffle=True,
        target_size=(im_height, im_width),
        class_mode='categorical'
    )
    total_train3 = train_data_gen3.n

    # 保存类别索引映射
    class_indices1 = train_data_gen1.class_indices
    class_indices2 = train_data_gen2.class_indices
    class_indices3 = train_data_gen3.class_indices

    inverse_dict1 = dict((val, key) for key, val in class_indices1.items())
    json_str1 = json.dumps(inverse_dict1, indent=4)
    inverse_dict2 = dict((val, key) for key, val in class_indices2.items())
    json_str2 = json.dumps(inverse_dict2, indent=4)
    inverse_dict3 = dict((val, key) for key, val in class_indices3.items())
    json_str3= json.dumps(inverse_dict3, indent=4)

    with open('class_indices1.json', 'w') as json_file:
        json_file.write(json.dumps(json_str1, indent=4))

    with open('class_indices2.json', 'w') as json_file:
        json_file.write(json.dumps(json_str2, indent=4))

    with open('class_indices3.json', 'w') as json_file:
        json_file.write(json.dumps(json_str3, indent=4))

    print("using {} images for training dataset 1.".format(total_train1))
    print("using {} images for training dataset 2.".format(total_train2))
    print("using {} images for training dataset 3.".format(total_train3))

    # 导入模型并查看模型信息
    model =create_model(im_height, im_width, num_class)
    model.summary()
    # 逻辑斯谛损失函数的计算函数

    # 对模型进行汇编，设置优化器、损失函数以及关注的数据
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adam优化器（自适应矩估计）学习率设为0.000
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 分类交叉熵损失函数，输出的logits需要经过激活函数的处
                  metrics=["accuracy"])  # 使用准确率评估当前训练模型的性能

    # 回调函数，保存权重文件
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='E:/2022grade/fxl/第一篇实验/save_weights/1.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='accuracy')]

    '''ModelCheckpoint()用于自动保存模型，保存性能最好的模型，且仅保存模型权重'''

    # 启用tf.function的即刻执行，而不是作为跟踪图函数运行，防止graph报错
    tf.config.experimental_run_functions_eagerly(True)
    combined_generator = zip(train_data_gen1, train_data_gen2, train_data_gen3)
    def combined_flow(combined_gen, batch_size):
        while True:
            # 注意这里简单地合并批次，实际应考虑平衡类别分布
            X_batch = []
            y_batch = []
            for gen in combined_gen:
                x, y = next(gen)
                X_batch.append(x)
                y_batch.append(y)
            # 假设每个批次大小相同，简单堆叠数据
            yield [np.concatenate(X_batch, axis=0), np.concatenate(y_batch, axis=0)]

    # 调整以使用新的数据生成方式
    combined_flow_generator = combined_flow(combined_generator, batch_size)

    # 修改fit调用来使用新的数据生成器
    history = model.fit(x=combined_flow_generator,
                        steps_per_epoch=int(np.ceil((total_train1 + total_train2 + total_train3) / batch_size)),
                        epochs=epochs,
                        callbacks=callbacks)

    # train_data_gen = concatenate([train_data_gen1, train_data_gen2, train_data_gen3], axis=3)
    # total_train = concatenate([total_train1, total_train2, total_train3], axis=3)
    # # 训练过程的一些信息保存在history中
    # history = model.fit(x=train_data_gen,# 输入训练数据
    #                     steps_per_epoch=total_train // batch_size,  # 每轮迭代的训练步骤数
    #                     epochs=epochs,  # 设置迭代次数
    #                     callbacks=callbacks)  # 返回权重文件

    # 返回权重文件

    # 获取到数据字典，保存了训练集的损失和准确率，验证集的损失和准确率
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    # 绘制损失函数与准确率变化折线图
    plt.figure()
    plt.plot(range(epochs), train_loss, color='red', label='train_loss')
    plt.plot(range(epochs), train_accuracy, color='blue', label='train_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss or accuracy')
    plt.show()


if __name__ == '__main__':
    main()
