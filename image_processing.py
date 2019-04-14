import tensorflow as tf
import numpy as np
import cv2 as cv

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.data import Dataset


VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

#前期图像转化
class ImageTurn(object):
    # 中值模糊  对椒盐噪声有很好的去燥效果
    def median_blur_demo(image):
        dst = cv.medianBlur(image, 5)
        return dst

    # 直方图均衡化
    def hisEqulColor(img):
        ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
        channels = cv.split(ycrcb)
        cv.equalizeHist(channels[0], channels[0])
        cv.merge(channels, ycrcb)
        cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
        return img

    # 双线性插值法
    def resize(src, new_size):
        dst_w, dst_h = new_size  # 目标图像宽高
        src_h, src_w = src.shape[:2]  # 源图像宽高
        if src_h == dst_h and src_w == dst_w:
            return src.copy()
        scale_x = float(src_w) / dst_w  # x缩放比例
        scale_y = float(src_h) / dst_h  # y缩放比例
        # 遍历目标图像，插值
        dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        for n in range(3):  # 对channel循环
            for dst_y in range(dst_h):  # 对height循环
                for dst_x in range(dst_w):  # 对width循环
                    # 目标在源上的坐标
                    src_x = (dst_x + 0.5) * scale_x - 0.5
                    src_y = (dst_y + 0.5) * scale_y - 0.5
                    # 计算在源图上四个近邻点的位置
                    src_x_0 = int(np.floor(src_x))
                    src_y_0 = int(np.floor(src_y))
                    src_x_1 = min(src_x_0 + 1, src_w - 1)
                    src_y_1 = min(src_y_0 + 1, src_h - 1)
                    # 双线性插值
                    value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
                    value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
                    dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
        return dst

# 把图片数据转化为三维矩阵
class ImageDataGenerator(object):
    def __init__(self, images, labels, batch_size, num_classes, image_format='jpg', shuffle=True):
        self.img_paths = images # [P1,P2]
        self.labels = labels # [1,2]
        self.data_size = len(self.labels)
        self.num_classes = num_classes
        self.image_format = image_format

        if shuffle:
            self._shuffle_lists()

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        data = data.map(self._parse_function_train)
        data = data.batch(batch_size)
        self.data = data

    # 打乱图片顺序
    def _shuffle_lists(self):
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    # 把图片生成三维数组，以及把标签转化为向量
    def _parse_function_train(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        if self.image_format == "jpg": # 增加图片类别区分
            img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        elif self.image_format == "png":
            img_decoded = tf.image.decode_png(img_string, channels=3)
        else:
            print("Error! Can't confirm the format of images!")
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr, one_hot