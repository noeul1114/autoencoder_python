import numpy as np
import random
import matplotlib
import PIL
import matplotlib.pyplot as plot

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8
)

seq = iaa.Sequential([
    iaa.Fliplr(0.1), # horizontal flips
    iaa.Crop(percent=(0, 0.05)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(1,
        iaa.GaussianBlur(sigma=(0, 0.1))
    ),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.9, 1.1), per_channel=0.1),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
], random_order=True) # apply augmenters in random order

seq2 = iaa.Sequential([
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(1,
        iaa.AverageBlur(k=2)
    ),
], random_order=True) # apply augmenters in random order


def weights_restart_3D_RGB(input_arr):
    x = input_arr
    for m in range(x.shape[1]):
        for n in range(x.shape[0]):
            count = 0
            for h in range(x.shape[2]):
                if x[n][m][h] == 0:
                    count += 1
            if count > 1:
                for h in range(x.shape[2]):
                    x[n][m][h] = random.random() * 2 - 1

    return x


def weights_restart_3D_mask(input_arr, mask_size, *bias):
    x = input_arr

    for i in range(x.shape[1] // mask_size):
        for j in range(x.shape[0] // mask_size):
            count = 0
            for h in range(x.shape[2]):
                for m in range(mask_size):
                    for n in range(mask_size):
                        if  x[mask_size * j + n][mask_size * i + m][h] == -1 \
                                or x[mask_size*j + n][mask_size*i+m][h] == 1:
                            count +=1
                # if count >= (mask_size**2)-2:
            if count >= 1:
                for h in range(x.shape[2]):
                    for m in range(mask_size):
                        for n in range(mask_size):
                            x[mask_size * j + n][mask_size * i + m][h] = random.random()
                            try:
                                bias[mask_size * j + n][mask_size * i + m][h] = 0
                            except:
                                pass
                            try:
                                bias[j][i][h] = 0
                            except:
                                pass

    return x



def weights_restart_4D_mask(input_arr, mask_size, *bias):
    x = input_arr
    for fc in range(x.shape[3]):
        for i in range(x.shape[1] // mask_size):
            for j in range(x.shape[0] // mask_size):
                count = 0
                for h in range(x.shape[2]):
                    for m in range(mask_size):
                        for n in range(mask_size):
                            if  x[mask_size * j + n][mask_size * i + m][h][fc] == -1 \
                                    or x[mask_size*j + n][mask_size*i+m][h][fc] == 1:
                                count +=1
                    # if count >= (mask_size**2)-2:
                if count >= 1:
                    for h in range(x.shape[2]):
                        for m in range(mask_size):
                            for n in range(mask_size):
                                x[mask_size * j + n][mask_size * i + m][h][fc] = random.random()
    return x


def weights_restart_3D_mask_101(input_arr, mask_size, *bias):
    x = input_arr

    for i in range(x.shape[1] // mask_size):
        for j in range(x.shape[0] // mask_size):
            count = 0
            for h in range(x.shape[2]):
                for m in range(mask_size):
                    for n in range(mask_size):
                        if  x[mask_size * j + n][mask_size * i + m][h] == 0 \
                                or x[mask_size * j + n][mask_size * i + m][h] == -1 \
                                or x[mask_size*j + n][mask_size*i+m][h] == 1:
                            count +=1
                # if count >= (mask_size**2)-2:
            if count >= 1:
                for h in range(x.shape[2]):
                    for m in range(mask_size):
                        for n in range(mask_size):
                            x[mask_size * j + n][mask_size * i + m][h] = random.random()
                            try:
                                bias[mask_size * j + n][mask_size * i + m][h] = 0
                            except:
                                pass
                            try:
                                bias[j][i][h] = 0
                            except:
                                pass

    return x

def weights_average_filter_3D_mask(input_arr, mask_size , *bias):
    x = input_arr

    for i in range(x.shape[1] // mask_size):
        for j in range(x.shape[0] // mask_size):
            count = 0
            for h in range(x.shape[2]):
                for m in range(mask_size):
                    for n in range(mask_size):
                        if x[mask_size * j + n][mask_size * i + m][h] == -1 \
                            or x[mask_size * j + n][mask_size * i + m][h] == 0 \
                                or x[mask_size * j + n][mask_size * i + m][h] == 1:
                            count += 1
                # if count >= (mask_size**2)-2:
            if count >= 1:
                for h in range(x.shape[2]):
                    for m in range(mask_size):
                        for n in range(mask_size):
                            space = 1
                            temp = 0
                            edge = 0
                            try:
                                temp += x[mask_size * j + n +space][mask_size * i + m][h]
                                edge += 1
                            except:
                                pass
                            try:
                                temp += x[mask_size * j + n -space][mask_size * i + m][h]
                                edge += 1
                            except:
                                pass
                            try:
                                temp += x[mask_size * j + n][mask_size * i + m - space][h]
                                edge += 1
                            except:
                                pass
                            try:
                                temp += x[mask_size * j + n][mask_size * i + m + space ][h]
                                edge += 1
                            except:
                                pass

                            x[mask_size * j + n][mask_size * i + m][h] = (((random.random())) + ((temp) /( edge)))/2

                            bias_detected = 0
                            try:
                                bias[mask_size * j + n][mask_size * i + m][h] = 0
                                bias_detected += 1
                            except:
                                pass
                            if bias_detected == 0:
                                try:
                                    bias[j][i][h] = 0
                                    bias_detected += 1
                                except:
                                    pass
                            if bias_detected == 0 :
                                try:
                                    bias[h] = 0
                                except:
                                    pass

    return x



def dropout_3D_weights(input_arr, rate, mask_size):
    x = input_arr
    for i in range(x.shape[0]):
        if x[i] > 15:
            x[i] = 15
        elif x[i] < -16:
            x[i] = -16
    return x

def bias_clipping_1616_1D(input_arr):
    x = input_arr
    for i in range(x.shape[0]):
        if x[i] > 15:
            x[i] = 15
        elif x[i] < -16:
            x[i] = -16
    return x


def bias_clipping_101_3D(input_arr):
    x = input_arr
    for h in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[h][i][j] >= 0.5:
                    x[h][i][j] = 1
                elif x[h][i][j] <= -0.5:
                    x[h][i][j] = -1
                elif x[h][i][j] < 0.5 and x[h][i][j] >-0.5:
                    x[h][i][j] = 0
    return x


def bias_clipping_11_3D(input_arr):
    x = input_arr
    for h in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[h][i][j] >= 1:
                    x[h][i][j] = 1
                elif x[h][i][j] <= -1:
                    x[h][i][j] = -1
    return x

def bias_clipping_128128_3D(input_arr):
    x = input_arr
    for h in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[h][i][j] >= 255:
                    x[h][i][j] = 255
                elif x[h][i][j] <= -256:
                    x[h][i][j] = -256
    return x

def weight_clipping_11_2D(input_arr):
    x = input_arr
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > 1:
                x[i][j] = 1
            elif x[i][j] < -1:
                x[i][j] = -1
    return x


def weight_clipping_1616_2D(input_arr):
    x = input_arr
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > 15:
                x[i][j] = 15
            elif x[i][j] < -16:
                x[i][j] = -16
    return x



def weight_clipping_1616_3D(input_arr):
    x = input_arr
    for h in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[h][i][j] > 15:
                    x[h][i][j] = 15
                elif x[h][i][j] < -16:
                    x[h][i][j] = -16
    return x


def weight_clipping_11_3D(input_arr):
    x = input_arr
    for h in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[h][i][j] > 1:
                    x[h][i][j] = 1
                elif x[h][i][j] < -1:
                    x[h][i][j] = -1
    return x


def weight_clipping_101_3D(input_arr):
    x = input_arr
    for h in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[h][i][j] >= 0.5:
                    x[h][i][j] = 1
                elif x[h][i][j] <= -0.5:
                    x[h][i][j] = -1
                elif x[h][i][j] < 0.5 and x[h][i][j] >-0.5:
                    x[h][i][j] = 0
    return x


def weight_clipping_1616_3D(input_arr):
    x = input_arr
    for h in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[h][i][j] > 15:
                    x[h][i][j] = 15
                elif x[h][i][j] < -16:
                    x[h][i][j] = -16
    return x

def int8_clipping_mul(x, y):
    temp = 0;
    temp = int(x) * int(y)
    if temp > 127:
        return 127
    elif temp < -128:
        return -128
    else:
        return temp

def casting_mul(x,y):
    temp = 0
    temp = int(x) * int(y)
    return temp

def int8_clipping_add(x, y):
    temp = 0;
    temp = int(x) + int(y)
    if temp > 127:
        return 127
    elif temp < -128:
        return -128
    else:
        return temp


def forward(x, w, b):
    return w * x + b


def relu(x):
    if x >= 0:
        return x
    else:
        return 0

def relu_dropout(x, rate):
    if x >= 0:
        if random.random() > rate:
            return x
        else:
            return 0
    else:
        return 0


def relu_prime(x):
    if x >= 0:
        return 1
    else:
        return 0


#
# LR = 0.001
#
#
# x = 15
#
# w = -1
#
# b = 1
#
# set = (w, b)
#
# ans = 99
#
# feature = forward(x, w, b)
# active = relu(feature)
#
# before = active
#
# before_loss = 20 * np.log10(np.square(ans - active))
#
# for i in range(1000):
#     feature = forward(x, w, b)
#     active = relu(feature)
#
#     dH = active - ans
#     loss = 20 * np.log10(np.square(ans - active))
#     print(loss)
#
#     dW = dH * relu_prime(feature) * x
#     dB = dH * relu_prime(feature)
#
#     w = w - LR * dW
#     b = b - LR * dW
#
# feature = forward(x, w, b)
# active = relu(feature)
#
# after = active
#
# after_loss = 20 * np.log10(np.square(ans - active))
#
# print("\n\n")
# print("Before active Value = {}".format(before))
# print("After active Value = {}".format(after))
# print("=======================================")
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))


# LR = 0.003
#
#
# x = 15
#
# w = random.randint(-1000, 1000)
#
# b = random.randint(-1000, 1000)
#
#
# ans = 99
#
# feature = forward(x, w, b)
# active = relu(feature)
#
# before = active
#
# before_loss = 20 * np.log10(np.square(ans - active))
#
# for i in range(1000):
#     feature = forward(x, w, b)
#     active = relu(feature)
#
#     dH = active - ans
#     loss = 20 * np.log10(np.square(ans - active))
#     print(loss)
#
#     dW = dH * relu_prime(feature) * x
#     dB = dH * relu_prime(feature)
#
#     w = w - LR * dW
#     b = b - LR * dW
#
# feature = forward(x, w, b)
# active = relu(feature)
#
# after = active
#
# after_loss = 20 * np.log10(np.square(ans - active))
#
# print("\n\n")
# print("Before active Value = {}".format(before))
# print("After active Value = {}".format(after))
# print("=======================================")
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))
# print("\nFinal Weight = {}, Final Bias = {}".format(w, b))

#
# LR = 0.0000003
#
# mu, sigma = 128, 40  # mean and standard deviation
# s = np.random.normal(mu, sigma, 9)
# s = s.reshape([3, 3])
# s = s.astype('int8')
#
# w = np.random.randint(-1, 1, 4)
# w = w.reshape([2, 2])
# w = w.astype('int8')
#
# b = np.random.randint(-1, 1, 1)
# b = b.astype('int8')
#
# ae1 = np.random.randint(-1, 1, 8)
# ae1 = ae1.reshape([2,2,2])
# ae1 = ae1.astype('int8')
#
# ae_b1 = np.random.randint(-1, 1, 2)
# ae_b1 = ae_b1.astype('int8')
#
# ae2 = np.random.randint(-1, 1, 8)
# ae2 = ae2.reshape([2,2,2])
# ae2 = ae2.astype('int8')
#
# ae_b2 = np.random.randint(-5, 5, 2)
# ae_b2 = ae_b2.astype('int8')
#
# ww = np.random.randint(-1, 1, 4)
# ww = ww.reshape([2, 2])
# ww = ww.astype('int8')
#
# bb = np.random.randint(-1, 1, 1)
# bb = bb.astype('int8')
#
# ##### initializing weights and biases
#
# for epoch in range(1000):
#     feature1 = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     feature1[x][y] += casting_mul(s[x+m][y+n] , w[m][n])
#
#     feature1 = feature1 + b[0]
#
#     active1 = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             active1[x][y] = relu(feature1[x][y])
#
#     ae_feature1 = np.zeros([2])
#
#     for h in range(2):
#         for x in range(2):
#             for y in range(2):
#                 for m in range(2):
#                     for n in range(2):
#                         ae_feature1[h] += casting_mul(active1[m][n] , ae1[h][x][y])
#
#         ae_feature1[h] = ae_feature1[h] + ae_b1[h]
#
#     ae_active1 = np.zeros([2])
#
#     for h in range(2):
#         ae_active1[h] = relu(ae_feature1[h])
#
#     ae_feature2 = np.zeros([2,2])
#
#     for h in range(2):
#         for x in range(2):
#             for y in range(2):
#                 ae_feature2[x][y] += casting_mul(ae_active1[h] , ae2[h][x][y])
#         ae_feature2 = ae_feature2 + ae_b2[h]
#
#     ae_active2 = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             ae_active2[x][y] = relu(ae_feature2[x][y])
#
#     # Transpose plane
#
#     TransposePlane = np.zeros([4,3,3])
#
#     for x in range(2):
#         for y in range(2):
#             TransposePlane[0][x][y] = ww[x][y]
#             TransposePlane[1][x][y +1] = ww[x][y]
#             TransposePlane[2][x+1][y] = ww[x][y]
#             TransposePlane[3][x + 1][y + 1] = ww[x][y]
#
#     ae_active2 = ae_active2.reshape([4])
#
#     feature2 = np.zeros([3,3])
#
#     for x in range(3):
#         for y in range(3):
#             for h in range(4):
#                 feature2[x][y] += casting_mul(TransposePlane[h][x][y], ae_active2[h])
#             feature2[x][y] = feature2[x][y] + bb[0]
#
#     active2 = np.zeros([3,3])
#
#     for x in range(3):
#         for y in range(3):
#             active2[x][y] = relu(feature2[x][y])
#
#     before = active2
#
#     # print(s)
#     # print("\n\n")
#     # print(before)
#
#     dH = active2 - s
#
#     loss = 20 * np.log10(np.sum(np.square(s - active2)))
#
#     print(loss)
#
#     dWW = np.zeros([4])
#     dBB = np.zeros([1])
#
#     for x in range(2):
#         for y in range(2):
#             dWW[0] += dH[x][y] * relu_prime(feature2[x][y]) * ae_active2[0]
#             dWW[1] += dH[x][y+1] * relu_prime(feature2[x][y+1]) * ae_active2[1]
#             dWW[2] += dH[x+1][y] * relu_prime(feature2[x+1][y]) * ae_active2[2]
#             dWW[3] += dH[x+1][y+1] * relu_prime(feature2[x+1][y+1]) * ae_active2[3]
#
#     for x in range(3):
#         for y in range(3):
#             dBB += dH[x][y] * relu_prime(feature2[x][y])
#
#     dBB = dBB / 9
#     dWW = dWW.reshape([2,2])
#
#     dAE2 = np.zeros([2,2,2])
#     dAE_B2 = np.zeros([2])
#
#     for h in range(2):
#         for x in range(2):
#             for y in range(2):
#                 dAE2[h][x][y] = dWW[x][y] * relu_prime(ae_feature2[x][y]) * ae_active1[h]
#
#     for h in range(2):
#         for x in range(2):
#             for y in range(2):
#                 dAE_B2[h] += dWW[x][y] * relu_prime(ae_feature2[x][y])
#
#     dAE_B2 = dAE_B2 / 4
#
#     dAE1 = np.zeros([2,2,2])
#     dAE_B1 = np.zeros([2])
#
#     dAE1_bottleneck = np.zeros([2])
#
#     for h in range(2):
#         for x in range(2):
#             for y in range(2):
#                 dAE1_bottleneck[h] += dAE2[h][x][y]
#
#     for h in range(2):
#         for x in range(2):
#             for y in range(2):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE1[h][x][y] += dAE1_bottleneck[h] * relu_prime(ae_feature1[h]) * active1[m][n]
#
#     for h in range(2):
#         dAE_B1[h] += dAE1_bottleneck[h] * relu_prime(ae_feature1[h])
#
#     dW = np.zeros([2,2])
#     dW_temp = np.zeros([2,2])
#     dB = np.zeros([1])
#
#     for x in range(2):
#         for y in range(2):
#             for h in range(2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW_temp[x][y] += dAE1[h][m][n]
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     dW[x][y] += dW_temp[x][y] * relu_prime(feature1[x][y]) * s[x+m][y+n]
#
#     for x in range(2):
#         for y in range(2):
#             dB += dW_temp[x][y] * relu_prime(feature1[x][y])
#
#     dB = dB / 4
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ae1 = ae1 - LR * dAE1
#     ae_b1 = ae_b1 - LR * dAE_B1
#     ae2 = ae2 - LR * dAE2
#     ae_b2 = ae_b2 - LR * dAE_B2
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB

#
# LR = 0.0001
#
# mu, sigma = 128, 40  # mean and standard deviation
# s = np.random.normal(mu, sigma, 9)
# s = s.reshape([3, 3])
#
# w = np.random.randint(-1, 1, 4)
# w = w.reshape([2, 2])
#
# b = np.random.randint(-1, 1, 1)
#
# ae = np.random.randint(-1, 1, 2)
#
# ae_b = np.random.randint(-1, 1, 2)
#
# ww = np.random.randint(-1, 1, 4)
# ww = ww.reshape([2, 2])
#
# bb = np.random.randint(-1, 1, 1)
#
# ##### initializing weights and biases
#
# for epoch in range(1000):
#     feature1 = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     feature1[x][y] += casting_mul(s[x+m][y+n] , w[m][n])
#
#     feature1 = feature1 + b[0]
#
#     active1 = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             active1[x][y] = relu(feature1[x][y])
#
#     ae_feature = np.zeros([2, 2])
#     ae_active = np.zeros([2, 2])
#
#     for h in range(2):
#         for m in range(2):
#             for n in range(2):
#                 ae_feature[m][n] += ((active1[m][n] * ae[h]) + ae_b[h])
#
#     for m in range(2):
#         for n in range(2):
#             ae_active[m][n] = relu(ae_feature[m][n])
#
#     # Transpose plane
#
#     TransposePlane = np.zeros([4,3,3])
#
#     for m in range(2):
#         for n in range(2):
#             TransposePlane[0][m][n] = ww[m][n]
#             TransposePlane[1][m][n+1] = ww[m][n]
#             TransposePlane[2][m+1][n] = ww[m][n]
#             TransposePlane[3][m + 1][n + 1] = ww[m][n]
#
#     ae_active = ae_active.reshape([4])
#
#     feature2 = np.zeros([3,3])
#
#     for x in range(3):
#         for y in range(3):
#             for h in range(4):
#                 feature2[x][y] += casting_mul(TransposePlane[h][x][y], ae_active[h])
#             feature2[x][y] = feature2[x][y] + bb[0]
#
#     active2 = np.zeros([3,3])
#
#     for x in range(3):
#         for y in range(3):
#             active2[x][y] = relu(feature2[x][y])
#
#     before = active2
#
#     # print(s)
#     # print("\n\n")
#     # print(before)
#
#     dH = active2 - s
#     dH /= 9
#
#     loss = 20 * np.log10(np.sum(np.square(s - active2)))
#
#     print(loss)
#
#     dWW = np.zeros([4])
#     dBB = np.zeros([1])
#
#     for x in range(2):
#         for y in range(2):
#             dWW[0] += dH[x][y] * relu_prime(feature2[x][y]) * ae_active[0]
#             dWW[1] += dH[x][y+1] * relu_prime(feature2[x][y+1]) * ae_active[1]
#             dWW[2] += dH[x+1][y] * relu_prime(feature2[x+1][y]) * ae_active[2]
#             dWW[3] += dH[x+1][y+1] * relu_prime(feature2[x+1][y+1]) * ae_active[3]
#
#     dWW /= 4
#
#     for x in range(3):
#         for y in range(3):
#             dBB += dH[x][y] * relu_prime(feature2[x][y])
#
#     dBB = dBB / 9
#     dWW = dWW.reshape([2,2])
#
#     dAE = np.zeros([2])
#     dAE_B = np.zeros([2])
#
#     for h in range(2):
#         for m in range(2):
#             for n in range(2):
#                 dAE[h] += dWW[m][n] * relu_prime(ae_feature[m][n]) * active1[m][n]
#     dAE /= 4
#
#     for h in range(2):
#         for m in range(2):
#             for n in range(2):
#                 dAE_B[h] += dWW[m][n] * relu_prime(ae_feature[m][n])
#     dAE_B = dAE_B / 4
#
#     dW = np.zeros([2,2])
#     dB = np.zeros([1])
#
#     for h in range(2):
#         for x in range(2):
#             for y in range(2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[x][y] += dAE[h] * relu_prime(feature1[m][n]) * s[x + m][y + n]
#
#     dW /= 8
#
#     for m in range(2):
#         for n in range(2):
#             dB[0] += dAE[h] * relu_prime(feature1[m][n])
#     dB = dB / 4
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ae = ae - LR * dAE
#     ae_b = ae_b - LR * dAE_B
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB


#
# LR = 0.00001
#
#
# s = np.random.randint(0, 255, [2,2])
#
# w = np.random.randint(0,10,[2])
#
# b = np.random.randint(0,10,[2])
#
# set = (w, b)
#
# feature = np.zeros([2,2])
# active = np.zeros([2,2])
#
# for h in range(2):
#     for m in range(2):
#         for n in range(2):
#             feature[m][n] += s[m][n] * w[h] + b[h]
#
# for m in range(2):
#     for n in range(2):
#         active[m][n] = relu(feature[m][n])
#
# before = active
#
# before_loss = 20 * np.log10(np.square(np.sum(s - active)))
#
# for i in range(1000):
#
#     feature = np.zeros([2, 2])
#     active = np.zeros([2, 2])
#
#     for h in range(2):
#         for m in range(2):
#             for n in range(2):
#                 feature[m][n] += s[m][n] * w[h] + b[h]
#
#     for m in range(2):
#         for n in range(2):
#             active[m][n] = relu(feature[m][n])
#
#     dH = active - s
#     loss = 20 * np.log10(np.square(np.sum(s - active)))
#     print(loss)
#
#     dW = np.zeros([2])
#     dB = np.zeros([2])
#
#     for h in range(2):
#         for m in range(2):
#             for n in range(2):
#                 dW[h] += dH[m][n] * relu_prime(feature[m][n]) * s[m][n]
#     for h in range(2):
#         dB[h] += dH[m][n] * relu_prime(feature[m][n])
#     dB = dB / 4
#
#     w = w - LR * dW
#     b = b - LR * dW
#
# feature = np.zeros([2, 2])
# active = np.zeros([2, 2])
#
# for h in range(2):
#     for m in range(2):
#         for n in range(2):
#             feature[m][n] += s[m][n] * w[h] + b[h]
#
# for m in range(2):
#     for n in range(2):
#         active[m][n] = relu(feature[m][n])
#
# after = active
#
# after_loss = 20 * np.log10(np.square(np.sum(s - active)))
#
# print("\n\n")
# print("Answer == \n{}".format(s))
# print("Before active Value = \n{}".format(before))
# print("After active Value = \n{}".format(after))
# print("=======================================")
#
# print("w = \n {} \nb = {} \n".format(w, b))
# print("=======================================")
#
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))

#
# LR = 0.000001
#
#
# s = np.random.randint(0, 255, [3,3])
#
# w = np.random.randint(-1,1,[2,3,3])
#
# b = np.random.randint(-1,1,[2])
#
# ww = np.random.randint(-1,1,[2,3,3])
#
# bb = np.random.randint(-1,1,[3,3])
#
# w_b = w
# b_b = b
# ww_b = ww
# bb_b = bb
#
# feature1 = np.zeros([2])
# active1 = np.zeros([2])
#
# for h in range(2):
#     for m in range(3):
#         for n in range(3):
#             feature1[h] += s[m][n] * w[h][m][n]
#
# for h in range(2):
#     active1[h] = relu(feature1[h] + b[h])
#
# feature2 = np.zeros([3, 3])
# active2 = np.zeros([3, 3])
#
# for h in range(2):
#     for m in range(3):
#         for n in range(3):
#             feature2[m][n] += active1[h] * ww[h][m][n]
#
# for m in range(3):
#     for n in range(3):
#         active2[m][n] = relu(feature2[m][n] + bb[m][n])
#
# loss = np.zeros([3, 3])
#
# loss = (np.square(s - active2)) / 2
#
# print(np.sum(loss))
#
# before = active2
#
# before_loss = np.sum((np.square(s - active2)) / 2)
#
# for i in range(1000):
#
#     feature1 = np.zeros([2])
#     active1 = np.zeros([2])
#
#     for h in range(2):
#         for m in range(3):
#             for n in range(3):
#                 feature1[h] += s[m][n] * w[h][m][n]
#
#     for h in range(2):
#         active1[h] = relu(feature1[h] + b[h])
#
#     feature2 = np.zeros([3, 3])
#     active2 = np.zeros([3, 3])
#
#     for h in range(2):
#         for m in range(3):
#             for n in range(3):
#                 feature2[m][n] += active1[h] * ww[h][m][n]
#
#     for m in range(3):
#         for n in range(3):
#             active2[m][n] = relu(feature2[m][n] + bb[m][n])
#
#     loss = np.zeros([3,3])
#
#     # for m in range(3):
#     #     for n in range(3):
#     #         dH[m][n] = -(s[m][n] * np.log10(active2[m][n]) + (1 - s[m][n]) * np.log10(1 - active2[m][n]))
#
#     loss = (np.square(s - active2)) / 2
#
#     print(np.sum(loss))
#
#     dH = active2 - s
#
#     dWW = np.zeros([2,3,3])
#     dBB = np.zeros([3,3])
#
#     for h in range(2):
#         for m in range(3):
#             for n in range(3):
#                 dWW[h][m][n] = dH[m][n] * relu_prime(feature2[m][n]) * active1[h]
#
#     for m in range(3):
#         for n in range(3):
#             dBB[m][n] = dH[m][n] * relu_prime(feature2[m][n])
#
#     dHH = np.zeros([2])
#
#     for h in range(2):
#         for m in range(3):
#             for n in range(3):
#                 dHH[h] += dH[m][n] * relu_prime(feature2[m][n]) * ww[h][m][n]
#
#     dW = np.zeros([2,3,3])
#     dB = np.zeros([2])
#
#     for h in range(2):
#         for m in range(3):
#             for n in range(3):
#                 dW[h][m][n] = dHH[h] * relu_prime(feature1[h]) * s[m][n]
#
#     for h in range(2):
#         dB[h] = dHH[h] * relu_prime(feature1[h])
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     # w = weight_clipping_11_3D(w)
#     # ww = weight_clipping_11_3D(ww)
#
#
#
#
# after = active2
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# print("Answer == \n{}".format(s))
# print("Before active Value = \n{}".format(before))
# print("After active Value = \n{}".format(after))
# print("=======================================")
#
# print("w = \n {} \nb = {} \n".format(w, b))
# print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# print("ww = \n {} \nbb = {} \n".format(ww, bb))
# print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
#
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))


#
# LR = 0.0000018
#
#
# s = np.random.randint(0, 255, [3,3])
#
# w = np.random.randint(-1,1,[2,2])
#
# # print(w)
# #
# # print(w[1][0])
# # print(w[0][1])
#
# b = np.random.randint(-1,1,[1])
#
# ww = np.random.randint(-1,1,[2,2])
#
# bb = np.random.randint(-1,1,[1])
#
# w_b = w
# b_b = b
# ww_b = ww
# bb_b = bb
#
# feature1 = np.zeros([2,2])
# active1 = np.zeros([2,2])
#
# for x in range(2):
#     for y in range(2):
#         for m in range(2):
#             for n in range(2):
#                 feature1[x][y] += w[m][n] * s[y+m][x+n]
#
# for x in range(2):
#     for y in range(2):
#         active1[x][y] = relu(feature1[x][y] + b[0])
#
# feature2 = np.zeros([3, 3])
# active2 = np.zeros([3, 3])
#
# active1_reshape = active1.reshape([4])
#
# # Transpose plane
# TransposePlane = np.zeros([4,3,3])
#
# for x in range(2):
#     for y in range(2):
#         TransposePlane[0][x][y] = ww[x][y]
#         TransposePlane[1][x][y +1] = ww[x][y]
#         TransposePlane[2][x+1][y] = ww[x][y]
#         TransposePlane[3][x + 1][y + 1] = ww[x][y]
#
# for m in range(3):
#     for n in range(3):
#         for h in range(4):
#             feature2[m][n] += active1_reshape[h] * TransposePlane[h][m][n]
#
# for m in range(3):
#     for n in range(3):
#         active2[m][n] = relu(feature2[m][n] + bb[0])
#
# loss = np.zeros([3, 3])
#
# loss = (np.square(s - active2)) / 2
#
# print(np.sum(loss))
#
# before = active2
#
# before_loss = np.sum((np.square(s - active2)) / 2)
#
# for i in range(10000):
#
#     feature1 = np.zeros([2, 2])
#     active1 = np.zeros([2, 2])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     feature1[x][y] += w[m][n] * s[y + m][x + n]
#
#     for x in range(2):
#         for y in range(2):
#             active1[x][y] = relu(feature1[x][y] + b[0])
#
#     feature2 = np.zeros([3, 3])
#     active2 = np.zeros([3, 3])
#
#     active1_reshape = active1.reshape([4])
#
#     # Transpose plane
#     TransposePlane = np.zeros([4, 3, 3])
#
#     for x in range(2):
#         for y in range(2):
#             TransposePlane[0][x][y] = ww[x][y]
#             TransposePlane[1][x][y + 1] = ww[x][y]
#             TransposePlane[2][x + 1][y] = ww[x][y]
#             TransposePlane[3][x + 1][y + 1] = ww[x][y]
#
#     for m in range(3):
#         for n in range(3):
#             for h in range(4):
#                 feature2[m][n] += active1_reshape[h] * TransposePlane[h][m][n]
#
#     for m in range(3):
#         for n in range(3):
#             active2[m][n] = relu(feature2[m][n] + bb[0])
#
#     loss = (np.square(s - active2)) / 2
#
#     print(np.sum(loss))
#
#     dH = active2 - s
#
#     dWW = np.zeros([2,2])
#     dBB = np.zeros([1])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     dWW[x][y] += dH[x+m][y+n] * relu_prime(feature2[x+m][y+n]) * active1[m][n]
#
#     for x in range(3):
#         for y in range(3):
#             dBB[0] += dH[x][y] * relu_prime(feature2[x][y])
#     dBB = dBB
#
#     dHH = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     dHH[x][y] += dH[x+m][y+n] * relu_prime(feature2[x+m][y+n]) * ww[m][n]
#
#     dW = np.zeros([2,2])
#     dB = np.zeros([1])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     dW[x][y] += dHH[x][y] * relu_prime(feature1[x][y]) * s[x+m][y+n]
#
#     for x in range(2):
#         for y in range(2):
#             dB[0] += dHH[x][y] * relu_prime(feature1[x][y])
#     dB = dB
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#
# after = active2
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# print("Answer == \n{}".format(s))
# print("Before active Value = \n{}".format(before))
# print("After active Value = \n{}".format(after))
# print("=======================================")
#
# print("w = \n {} \nb = {} \n".format(w, b))
# print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# print("ww = \n {} \nbb = {} \n".format(ww, bb))
# print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
#
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))

#
# LR = 0.0000001
#
#
# mu, sigma = 0, 1  # mean and standard deviation
#
# # s == input (3x3)
# # w == encoder weight (2x2)
# # ww == decoder weight (2x2)
# # b, bb == biases
# s = np.random.randint(0, 255, [3,3])
# w = np.random.normal(mu,sigma,[2,2])
# b = np.random.normal(mu,sigma,[2,2])
#
# aw1 = np.random.normal(mu, sigma, [2,2,2])
# ab1 = np.random.normal(mu, sigma, [2,2])
#
# aw2 = np.random.normal(mu, sigma, [2,2,2])
# ab2 = np.random.normal(mu, sigma, [2,2])
#
#
# ww = np.random.normal(mu,sigma,[2,2])
# bb = np.random.normal(mu,sigma,[3,3])
#
# # temporary initial forward values
# w_b = w
# b_b = b
# ww_b = ww
# bb_b = bb
#
# # feature1 == feature map from encoder weight
# # active1 == activation map from encoder weight
#
# feature1 = np.zeros([2,2])
# active1 = np.zeros([2,2])
#
# for x in range(2):
#     for y in range(2):
#         for m in range(2):
#             for n in range(2):
#                 feature1[x][y] += w[m][n] * s[y+m][x+n]
#
# for x in range(2):
#     for y in range(2):
#         active1[x][y] = relu(feature1[x][y] + b[x][y])
#
# feature2 = np.zeros([3, 3])
# active2 = np.zeros([3, 3])
#
# # reshape for Transposed convolution
# # not essential but for my convenience
# active1_reshape = active1.reshape([4])
#
# # Transpose plane
# # For more information,
# # https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/
# # checkout for Transposed Convolution matrix
#
# # In the link, He constructed Transposed convolution matrix with (out width * out height, (mask size^2))
# # but I've constructed Transposed convolution matrix with (out width, out height, mask size^2)
# TransposePlane = np.zeros([4,3,3])
#
# for x in range(2):
#     for y in range(2):
#         TransposePlane[0][x][y] = ww[x][y]
#         TransposePlane[1][x][y +1] = ww[x][y]
#         TransposePlane[2][x+1][y] = ww[x][y]
#         TransposePlane[3][x + 1][y + 1] = ww[x][y]
#
# for m in range(3):
#     for n in range(3):
#         for h in range(4):
#             feature2[m][n] += active1_reshape[h] * TransposePlane[h][m][n]
#
#
#
# for m in range(3):
#     for n in range(3):
#         active2[m][n] = relu(feature2[m][n] + bb[m][n])
#
# loss = np.zeros([3, 3])
#
# # simple square mean loss
# loss = (np.square(s - active2)) / 9
#
# print(np.sum(loss))
#
# before = active2
#
# before_loss = np.sum((np.square(s - active2)) / 2)
#
# for i in range(10000):
#
#     feature1 = np.zeros([2, 2])
#     active1 = np.zeros([2, 2])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     feature1[x][y] += w[m][n] * s[y + m][x + n]
#
#     for x in range(2):
#         for y in range(2):
#             active1[x][y] = relu(feature1[x][y] + b[x][y])
#
#     feature2 = np.zeros([3, 3])
#     active2 = np.zeros([3, 3])
#
#     active1_reshape = active1.reshape([4])
#
#     # Transpose plane
#     TransposePlane = np.zeros([4, 3, 3])
#
#     for x in range(2):
#         for y in range(2):
#             TransposePlane[0][x][y] = ww[x][y]
#             TransposePlane[1][x][y + 1] = ww[x][y]
#             TransposePlane[2][x + 1][y] = ww[x][y]
#             TransposePlane[3][x + 1][y + 1] = ww[x][y]
#
#     for m in range(3):
#         for n in range(3):
#             for h in range(4):
#                 feature2[m][n] += active1_reshape[h] * TransposePlane[h][m][n]
#
#     for m in range(3):
#         for n in range(3):
#             active2[m][n] = relu(feature2[m][n] + bb[m][n])
#
#     loss = (np.square(s - active2)) / 9
#
#     print(np.sum(loss))
#
#     dH = active2 - s
#
#     dWW = np.zeros([2,2])
#     dBB = np.zeros([3,3])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     dWW[x][y] += dH[x+m][y+n] * relu_prime(feature2[x+m][y+n]) * active1[m][n]
#
#     for x in range(3):
#         for y in range(3):
#             dBB[x][y] += dH[x][y] * relu_prime(feature2[x][y])
#
#     dHH = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     dHH[x][y] += dH[x+m][y+n] * relu_prime(feature2[x+m][y+n]) * ww[m][n]
#
#     dW = np.zeros([2,2])
#     dB = np.zeros([2,2])
#
#     for x in range(2):
#         for y in range(2):
#             for m in range(2):
#                 for n in range(2):
#                     dW[x][y] += dHH[x][y] * relu_prime(feature1[x][y]) * s[x+m][y+n]
#
#     for x in range(2):
#         for y in range(2):
#             dB[x][y] += dHH[x][y] * relu_prime(feature1[x][y])
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     # w = weight_clipping_1616_2D(w)
#     # ww = weight_clipping_1616_2D(ww)
#
#     # b = bias_clipping_1616_1D(b)
#     # bb = bias_clipping_1616_1D(bb)
#
#
# after = active2
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# print("Answer == \n{}".format(s))
# print("Before active Value = \n{}".format(before))
# print("After active Value = \n{}".format(after))
# print("=======================================")
#
# print("w = \n {} \nb = {} \n".format(w, b))
# print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# print("ww = \n {} \nbb = {} \n".format(ww, bb))
# print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
#
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))
#


#
# LR = 1
#
# img = plot.imread("AKR20160511151000033_02_i.jpg")
#
# img2 = plot.imread("33b1383713b057fc6e3097cee5902d35.jpg")
#
# # plot.imshow(img)
# # plot.show()
#
# print(img)
#
# copy1 = img
# copy2 = img2
#
# size = img.shape
#
# height = img.shape[0]
# width = img.shape[1]
# RGB = img.shape[2]
#
# mu, sigma = 0, 1  # mean and standard deviation
#
# # s = np.random.randint(0, 255, [3,3])
#
# w = np.random.normal(mu,sigma,[height, width, RGB])
# b = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ww = np.random.normal(mu,sigma,[height, width, RGB])
# bb = np.random.normal(mu,sigma,[height, width, RGB])
#
# ww2 = np.random.normal(mu,sigma,[height, width, RGB])
# bb2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# w_b = w
# b_b = b
# ww_b = ww
# bb_b = bb
#
# feature1 = np.zeros([height//2, width//2, RGB])
# active1 = np.zeros([height//2, width//2, RGB])
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             for m in range(2):
#                 for n in range(2):
#                     feature1[y][x][h] += copy1[2*y+n][2*x+m][h] * w[2*y+n][2*x+m][h]
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
#
# feature2 = np.zeros([height, width, RGB])
# active2 = np.zeros([height, width, RGB])
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             for m in range(2):
#                 for n in range(2):
#                     feature2[2*y+n][2*x+m][h] += active2[y][x][h] * ww[2*y+n][2*x+m][h]
#
# for h in range(RGB):
#     for x in range(width):
#         for y in range(height):
#             active2[y][x][h] = relu(feature2[y][x][h] + bb[y][x][h])
#
# loss = np.sum((np.square(copy1 - active2)) / (width*height))
#
# print(np.sum(loss))
#
# plot.imshow(active2)
# plot.show()
#
# before = active2
#
# before_loss = loss
#
# for epoch in range(6):
#     if epoch % 2 ==0:
#         copy = copy1
#     else:
#         copy = copy2
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += copy[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
#
#     feature2 = np.zeros([height, width, RGB])
#     active2 = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[2 * y + n][2 * x + m][h] += active2[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 active2[y][x][h] = relu(feature2[y][x][h] + bb[y][x][h])
#
#     loss = np.sum((np.square(copy - active2))) / (width * height)
#
#     temp_out = np.square(copy - active2)
#
#     print(np.sum(loss))
#     dH = active2 - copy
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#     temp = active2.astype('uint8')
#
#     plot.imshow(active2.astype('uint8'))
#     plot.show()
#
#     dWW = np.zeros([height, width, RGB])
#     dBB = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dWW[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature2[2*y+n][2*x+m][h]) * active1[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dBB[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature2[2*y+n][2*x+m][h])
#
#     dHH = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature2[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]
#
#     dW = np.zeros([height, width, RGB])
#     dB = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[2*y+n][2*x+m][h] = dHH[y][x][h] * relu_prime(feature1[y][x][h]) * copy[2*y+n][2*x+m][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dB[y][x][h] = dHH[y][x][h] * relu_prime(feature1[y][x][h])
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     # w = weight_clipping_11_3D(w)
#     # ww = weight_clipping_11_3D(ww)
#     w = weight_clipping_101_3D(w)
#     ww = weight_clipping_101_3D(ww)
#
#
# after = active2
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# # print("Answer == \n{}".format(copy))
# # print("Before active Value = \n{}".format(before.astype('int')))
# # print("After active Value = \n{}".format(after.astype('int')))
# print("=======================================")
#
# plot.imshow(img)
# plot.show()
#
# plot.imshow(active2.astype('uint8'))
# plot.show()
#
# plot.imsave("Output.jpg", active2.astype('uint8'))
#
# # print("w = \n {} \nb = {} \n".format(w, b))
# # print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# # print("ww = \n {} \nbb = {} \n".format(ww, bb))
# # print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
#
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))


#
# LR = 0.1
#
# img = plot.imread("AKR20160511151000033_02_i.jpg")
#
# img2 = plot.imread("33b1383713b057fc6e3097cee5902d35.jpg")
#
# # plot.imshow(img)
# # plot.show()
#
# # print(img)
#
# copy1 = img
# copy2 = img2
#
# size = img.shape
#
# height = img.shape[0]
# width = img.shape[1]
# RGB = img.shape[2]
#
# mu, sigma = 0, 1  # mean and standard deviation
#
# # s = np.random.randint(0, 255, [3,3])
#
# w = np.random.normal(mu,sigma,[height, width, RGB])
# b = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ww1 = np.random.normal(mu,sigma,[height, width, RGB])
# bb1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# ww2 = np.random.normal(mu,sigma,[height, width, RGB])
# bb2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# w_b = w
# b_b = b
# ww_b = ww1
# bb_b = bb1
#
# feature1 = np.zeros([height//2, width//2, RGB])
# active1 = np.zeros([height//2, width//2, RGB])
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             for m in range(2):
#                 for n in range(2):
#                     feature1[y][x][h] += copy1[2*y+n][2*x+m][h] * w[2*y+n][2*x+m][h]
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
#
# feature2 = np.zeros([height, width, RGB])
# active2 = np.zeros([height, width, RGB])
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             for m in range(2):
#                 for n in range(2):
#                     feature2[2*y+n][2*x+m][h] += active2[y][x][h] * ww1[2*y+n][2*x+m][h]
#
# for h in range(RGB):
#     for x in range(width):
#         for y in range(height):
#             active2[y][x][h] = relu(feature2[y][x][h] + bb1[y][x][h])
#
# loss = np.sum((np.square(copy1 - active2)) / (width*height))
#
# print(np.sum(loss))
#
# plot.imshow(active2)
# plot.show()
#
# before = active2
#
# before_loss = loss
#
# for epoch in range(10):
#     print("epoch == {}".format(epoch))
#     if epoch % 2 ==0:
#         copy = copy1
#         ww = ww1
#         bb = bb1
#     else:
#         copy = copy2
#         ww = ww2
#         bb = bb2
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += copy[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
#
#     feature2 = np.zeros([height, width, RGB])
#     active2 = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[2 * y + n][2 * x + m][h] += active2[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 active2[y][x][h] = relu(feature2[y][x][h] + bb[y][x][h])
#
#     loss = np.sum((np.square(copy - active2))) / (width * height)
#
#     temp_out = np.square(copy - active2)
#
#     print(np.sum(loss))
#     dH = active2 - copy
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#     temp = active2.astype('uint8')
#
#     plot.imshow(active2.astype('uint8'))
#     plot.show()
#
#     dWW = np.zeros([height, width, RGB])
#     dBB = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dWW[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature2[2*y+n][2*x+m][h]) * active1[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dBB[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature2[2*y+n][2*x+m][h])
#
#     dHH = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature2[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]
#
#     dW = np.zeros([height, width, RGB])
#     dB = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[2*y+n][2*x+m][h] = dHH[y][x][h] * relu_prime(feature1[y][x][h]) * copy[2*y+n][2*x+m][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dB[y][x][h] = dHH[y][x][h] * relu_prime(feature1[y][x][h])
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     w = weight_clipping_11_3D(w)
#     ww = weight_clipping_11_3D(ww)
#     # w = weight_clipping_101_3D(w)
#     # ww = weight_clipping_101_3D(ww)
#
#     if epoch % 2 ==0:
#         ww1 = ww
#         bb1 = bb
#     else:
#         ww2 = ww
#         bb2 = bb
#
#
# for pp in range(4):
#     if pp == 0:
#         copy = copy1
#         ww = ww2
#         bb = bb2
#         print("first")
#     elif pp == 1:
#         copy = copy1
#         ww = ww1
#         bb = bb1
#     elif pp == 2:
#         copy = copy2
#         ww = ww1
#         bb = bb1
#     elif pp == 3:
#         copy = copy2
#         ww = ww2
#         bb = bb2
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += copy[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
#
#     feature2 = np.zeros([height, width, RGB])
#     active2 = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[2 * y + n][2 * x + m][h] += active2[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 active2[y][x][h] = relu(feature2[y][x][h] + bb[y][x][h])
#
#     loss = np.sum((np.square(copy - active2))) / (width * height)
#
#     temp_out = np.square(copy - active2)
#
#     print(np.sum(loss))
#     dH = active2 - copy
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#     temp = active2.astype('uint8')
#
#     plot.imshow(active2.astype('uint8'))
#     plot.show()
#
#
#
# after = active2
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# # print("Answer == \n{}".format(copy))
# # print("Before active Value = \n{}".format(before.astype('int')))
# # print("After active Value = \n{}".format(after.astype('int')))
# print("=======================================")
#
# plot.imshow(img)
# plot.show()
#
# plot.imshow(active2.astype('uint8'))
# plot.show()
#
# plot.imsave("Output.jpg", active2.astype('uint8'))
#
# # print("w = \n {} \nb = {} \n".format(w, b))
# # print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# # print("ww = \n {} \nbb = {} \n".format(ww, bb))
# # print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
#
# print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))



#
#
# LR = 1
#
# img = plot.imread("edit_AKR20160511151000033_02_i.jpg")
#
# img2 = plot.imread("edit_33b1383713b057fc6e3097cee5902d35.jpg")
#
# # plot.imshow(img)
# # plot.show()
#
# print(img)
#
# copy1 = img
# copy2 = img2
#
# size = img.shape
#
# height = img.shape[0]
# width = img.shape[1]
# RGB = img.shape[2]
#
# mu, sigma = 0, 1  # mean and standard deviation
#
# # s = np.random.randint(0, 255, [3,3])
#
# w = np.random.normal(mu,sigma,[height, width, RGB])
# b = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_w1 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b1 = np.random.normal(mu,sigma,[height//4, width//4, RGB])
#
# ae_w2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_ww2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_bb2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ww1 = np.random.normal(mu,sigma,[height, width, RGB])
# bb1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# ww2 = np.random.normal(mu,sigma,[height, width, RGB])
# bb2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# # w_b = w
# # b_b = b
# # ww_b = ww
# # bb_b = bb
# #
# # feature1 = np.zeros([height//2, width//2, RGB])
# # active1 = np.zeros([height//2, width//2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature1[y][x][h] += copy1[2*y+n][2*x+m][h] * w[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
# #
# # feature2 = np.zeros([height//4, width//4, RGB])
# # active2 = np.zeros([height//4, width//4, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature2[y][x][h] += active1[2*y+n][2*x+m][h] * ae_w1[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             active2[y][x][h] = relu(feature2[y][x][h] + ae_b1[y][x][h])
# #
# # feature3 = np.zeros([height // 2, width // 2, RGB])
# # active3 = np.zeros([height // 2, width // 2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature3[2*y+n][2*x+m][h] += active2[y][x][h] * ae_w2[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active3[y][x][h] = relu(feature3[y][x][h] + ae_b2[y][x][h])
# #
# #
# # feature4 = np.zeros([height, width, RGB])
# # active4 = np.zeros([height, width, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature4[2*y+n][2*x+m][h] += active3[y][x][h] * ww[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width):
# #         for y in range(height):
# #             active4[y][x][h] = relu(feature4[y][x][h] + bb[y][x][h])
# #
# # loss = np.sum((np.square(copy1 - active4)) / (width*height))
# #
# # print(np.sum(loss))
# #
# # plot.imshow(active4)
# # plot.show()
# #
# # before = active4
# #
# # before_loss = loss
#
# for epoch in range(500):
#     print("epoch == {}".format(epoch))
#     if epoch % 2 == 0:
#         copy = copy1
#         ae_dw2 = ae_w2
#         ae_db2 = ae_b2
#         ww = ww1
#         bb = bb1
#     else:
#         copy = copy2
#         ae_dw2 = ae_ww2
#         ae_db2 = ae_bb2
#         ww = ww2
#         bb = bb2
#
#     rate = 0.0
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += copy1[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)
#
#     feature2 = np.zeros([height // 4, width // 4, RGB])
#     active2 = np.zeros([height // 4, width // 4, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)
#
#     feature3 = np.zeros([height // 2, width // 2, RGB])
#     active3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature3[2 * y + n][2 * x + m][h] += active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)
#
#     feature4 = np.zeros([height, width, RGB])
#     active4 = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)
#
#     loss = np.sum((np.square(copy1 - active4)) / (width * height))
#
#     print(np.sum(loss))
#
#     dH = active4 - copy
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#
#     plot.imshow(active1.astype('uint8'))
#     plot.show()
#     # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
#     plot.imshow(active2.astype('uint8'))
#     plot.show()
#     # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
#     plot.imshow(active3.astype('uint8'))
#     plot.show()
#     # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))
#     plot.imshow(active4.astype('uint8'))
#     plot.show()
#     plot.imsave("epoch_{}_LR_{}_DROPOUT_{}.jpg".format(epoch, LR, rate), active4.astype('uint8'))
#
#     dWW = np.zeros([height, width, RGB])
#     dBB = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dWW[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * active3[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dBB[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h])
#
#     dHH = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]
#
#     dAE_W2 = np.zeros([height//2, width//2, RGB])
#     dAE_B2 = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * active2[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h])
#
#     dHH2 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * ae_dw2[2*y+n][2*x+m][h]
#
#     dAE_W1 = np.zeros([height//2, width//2, RGB])
#     dAE_B1 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W1[2*y+n][2*x+m][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * active1[2*y+n][2*x+m][h]
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B1[y][x][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h])
#
#     dHH3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH3[2*y+n][2*x+m][h] += dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * ae_w1[2*y+n][2*x+m][h]
#
#     dW = np.zeros([height, width, RGB])
#     dB = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[2*y+n][2*x+m][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h]) * copy[2*y+n][2*x+m][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dB[y][x][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h])
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ae_w1 = ae_w1 - LR * dAE_W1
#     ae_b1 = ae_b1 - LR * dAE_B1
#
#     ae_dw2 = ae_dw2 - LR * dAE_W2
#     ae_db2 = ae_db2 - LR * dAE_B2
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     # w = weight_clipping_11_3D(w)
#     # ww = weight_clipping_11_3D(ww)
#     w = weight_clipping_11_3D(w)
#     ww = weight_clipping_11_3D(ww)
#
#     ae_w1 = weight_clipping_11_3D(ae_w1)
#     ae_w2 = weight_clipping_11_3D(ae_dw2)
#
#     # b = bias_clipping_1616_3D(b)
#     # bb = bias_clipping_1616_3D(bb)
#     #
#     # ae_b1 = bias_clipping_1616_3D(ae_b1)
#     # ae_b2 = bias_clipping_1616_3D(ae_b2)
#
#     if epoch % 2 == 0:
#         copy = copy1
#         ae_w2 = ae_dw2
#         ae_b2 = ae_db2
#         ww1 = ww
#         bb1 = bb
#     else:
#         copy = copy2
#         ae_ww2 = ae_dw2
#         ae_bb2 = ae_db2
#         ww2 = ww
#         bb2 = bb
#
# after = active2
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# # print("Answer == \n{}".format(copy))
# # print("Before active Value = \n{}".format(before.astype('int')))
# # print("After active Value = \n{}".format(after.astype('int')))
# print("=======================================")
#
# plot.imshow(img)
# plot.show()
#
# plot.imshow(active4.astype('uint8'))
# plot.show()
#
# plot.imsave("Output.jpg", active4.astype('uint8'))
#
# # print("w = \n {} \nb = {} \n".format(w, b))
# # print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# # print("ww = \n {} \nbb = {} \n".format(ww, bb))
# # print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
# #
# # print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))
#


#
#
# LR = 1
#
# img = plot.imread("edit_AKR20160511151000033_02_i.jpg")
#
# img2 = plot.imread("edit_33b1383713b057fc6e3097cee5902d35.jpg")
#
# # plot.imshow(img)
# # plot.show()
#
# print(img)
#
# copy1 = img
# copy2 = img2
#
# size = img.shape
#
# height = img.shape[0]
# width = img.shape[1]
# RGB = img.shape[2]
#
# mu, sigma = 0, 1  # mean and standard deviation
#
# # s = np.random.randint(0, 255, [3,3])
#
#
# RGB_ew = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_eb = np.random.normal(mu,sigma,[height, width, 1])
#
# w = np.random.normal(mu,sigma,[height, width, RGB])
# b = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_w1 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b1 = np.random.normal(mu,sigma,[height//4, width//4, RGB])
#
# ae_w2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_ww2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_bb2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ww1 = np.random.normal(mu,sigma,[height, width, RGB])
# bb1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# ww2 = np.random.normal(mu,sigma,[height, width, RGB])
# bb2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# RGB_dw1 = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_db1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# RGB_dw2 = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_db2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# # w_b = w
# # b_b = b
# # ww_b = ww
# # bb_b = bb
# #
# # feature1 = np.zeros([height//2, width//2, RGB])
# # active1 = np.zeros([height//2, width//2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature1[y][x][h] += copy1[2*y+n][2*x+m][h] * w[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
# #
# # feature2 = np.zeros([height//4, width//4, RGB])
# # active2 = np.zeros([height//4, width//4, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature2[y][x][h] += active1[2*y+n][2*x+m][h] * ae_w1[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             active2[y][x][h] = relu(feature2[y][x][h] + ae_b1[y][x][h])
# #
# # feature3 = np.zeros([height // 2, width // 2, RGB])
# # active3 = np.zeros([height // 2, width // 2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature3[2*y+n][2*x+m][h] += active2[y][x][h] * ae_w2[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active3[y][x][h] = relu(feature3[y][x][h] + ae_b2[y][x][h])
# #
# #
# # feature4 = np.zeros([height, width, RGB])
# # active4 = np.zeros([height, width, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature4[2*y+n][2*x+m][h] += active3[y][x][h] * ww[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width):
# #         for y in range(height):
# #             active4[y][x][h] = relu(feature4[y][x][h] + bb[y][x][h])
# #
# # loss = np.sum((np.square(copy1 - active4)) / (width*height))
# #
# # print(np.sum(loss))
# #
# # plot.imshow(active4)
# # plot.show()
# #
# # before = active4
# #
# # before_loss = loss
#
# for epoch in range(500):
#     print("epoch == {}".format(epoch))
#     if epoch % 2 == 0:
#         copy = seq.augment_image(copy1)
#         ae_dw2 = ae_w2
#         ae_db2 = ae_b2
#         ww = ww1
#         bb = bb1
#         RGB_dw = RGB_dw1
#         RGB_db = RGB_db1
#     else:
#         copy = seq.augment_image(copy2)
#         ae_dw2 = ae_ww2
#         ae_db2 = ae_bb2
#         ww = ww2
#         bb = bb2
#         RGB_dw = RGB_dw2
#         RGB_db = RGB_db2
#
#     # plot.imshow(copy)
#     # plot.show()
#
#     rate = 0.0
#
#     rgb_feature1 = np.zeros([height, width, 1])
#     rgb_active1 = np.zeros([height, width, 1])
#
#     for x in range(width):
#         for y in range(height):
#             for h in range(RGB):
#                 rgb_feature1[y][x][0] += copy[y][x][h] * RGB_ew[y][x][h]
#
#     for x in range(width):
#         for y in range(height):
#             rgb_active1[y][x][0] = relu(rgb_feature1[y][x][0])
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += rgb_active1[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)
#
#     feature2 = np.zeros([height // 4, width // 4, RGB])
#     active2 = np.zeros([height // 4, width // 4, RGB])
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)
#
#     feature3 = np.zeros([height // 2, width // 2, RGB])
#     active3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature3[2 * y + n][2 * x + m][h] += active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)
#
#     feature4 = np.zeros([height, width, RGB])
#     active4 = np.zeros([height, width, RGB])
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width):
#             for y in range(height):
#                 active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)
#
#     rgb_feature2 =np.zeros([height, width, RGB])
#     rgb_active2 =np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 rgb_feature2[y][x][h] += active4[y][x][0] * RGB_dw[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 rgb_active2[y][x][h] = relu_dropout(rgb_feature2[y][x][h] + RGB_db[y][x][h], rate)
#
#     loss = np.sum((np.square(copy - rgb_active2)) / (width * height))
#
#     print(np.sum(loss))
#
#     dH = rgb_active2 - copy
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#
#     # plot.imshow(active1.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
#     # plot.imshow(active2.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
#     # plot.imshow(active3.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))
#     # plot.imshow(active4.astype('uint8'))
#     # plot.show()
#     plot.imshow(rgb_active2.astype('uint8'))
#     plot.show()
#     plot.imsave("epoch_{}_LR_{}_DROPOUT_{}.jpg".format(epoch, LR, rate), rgb_active2.astype('uint8'))
#
#     dRGB_dDW = np.zeros([height, width, RGB])
#     dRGB_dDB = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dDW[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0]) * active4[y][x][0]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dDB[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0])
#
#     dH_decode_RGB = np.zeros([height, width, 1])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dH_decode_RGB[y][x][0] += dH[y][x][h] * relu_prime(rgb_feature2[y][x][h]) * RGB_dw[y][x][h]
#
#
#     dWW = np.zeros([height, width, RGB])
#     dBB = np.zeros([height, width, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dWW[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * active3[y][x][h]
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dBB[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h])
#
#     dHH = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]
#
#     dAE_W2 = np.zeros([height//2, width//2, RGB])
#     dAE_B2 = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * active2[y][x][h]
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h])
#
#     dHH2 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * ae_dw2[2*y+n][2*x+m][h]
#
#     dAE_W1 = np.zeros([height//2, width//2, RGB])
#     dAE_B1 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W1[2*y+n][2*x+m][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * active1[2*y+n][2*x+m][h]
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B1[y][x][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h])
#
#     dHH3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH3[2*y+n][2*x+m][h] += dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * ae_w1[2*y+n][2*x+m][h]
#
#     dW = np.zeros([height, width, RGB])
#     dB = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[2*y+n][2*x+m][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h]) * rgb_active1[2*y+n][2*x+m][h]
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dB[y][x][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h])
#
#     dRGB_dEW = np.zeros([height, width, RGB])
#     dRGB_dEB = np.zeros([height, width, 1])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dEW[y][x][h] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0]) * copy[y][x][h]
#
#     for h in range(1):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dEB[y][x][0] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0])
#
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ae_w1 = ae_w1 - LR * dAE_W1
#     ae_b1 = ae_b1 - LR * dAE_B1
#
#     ae_dw2 = ae_dw2 - LR * dAE_W2
#     ae_db2 = ae_db2 - LR * dAE_B2
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     RGB_ew = RGB_ew - LR * dRGB_dEW
#     RGB_eb = RGB_eb - LR * dRGB_dEB
#
#     RGB_dw = RGB_dw - LR * dRGB_dDW
#     RGB_db = RGB_db - LR * dRGB_dDB
#
#     # w = weight_clipping_11_3D(w)
#     # ww = weight_clipping_11_3D(ww)
#     w = weight_clipping_11_3D(w)
#     ww = weight_clipping_11_3D(ww)
#
#     ae_w1 = weight_clipping_11_3D(ae_w1)
#     ae_w2 = weight_clipping_11_3D(ae_dw2)
#
#     RGB_ew = weight_clipping_11_3D(RGB_ew)
#     RGB_dw = weight_clipping_11_3D(RGB_dw)
#
#     # b = bias_clipping_1616_3D(b)
#     # bb = bias_clipping_1616_3D(bb)
#     #
#     # ae_b1 = bias_clipping_1616_3D(ae_b1)
#     # ae_b2 = bias_clipping_1616_3D(ae_b2)
#
#     if epoch % 2 == 0:
#         ae_w2 = ae_dw2
#         ae_b2 = ae_db2
#         ww1 = ww
#         bb1 = bb
#
#         RGB_dw1 = RGB_dw
#         RGB_db1 = RGB_db
#     else:
#         ae_ww2 = ae_dw2
#         ae_bb2 = ae_db2
#         ww2 = ww
#         bb2 = bb
#
#         RGB_dw2 = RGB_dw
#         RGB_db2 = RGB_db
#
# after = active4
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# # print("Answer == \n{}".format(copy))
# # print("Before active Value = \n{}".format(before.astype('int')))
# # print("After active Value = \n{}".format(after.astype('int')))
# print("=======================================")
#
# plot.imshow(img)
# plot.show()
#
# plot.imshow(active4.astype('uint8'))
# plot.show()
#
# plot.imsave("Output.jpg", active4.astype('uint8'))
#
# # print("w = \n {} \nb = {} \n".format(w, b))
# # print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# # print("ww = \n {} \nbb = {} \n".format(ww, bb))
# # print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
# #
# # print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))







#
# LR = 1
#
# img = plot.imread("edit_AKR20160511151000033_02_i.jpg")
#
# img2 = plot.imread("edit_33b1383713b057fc6e3097cee5902d35.jpg")
#
# # plot.imshow(img)
# # plot.show()
#
# # print(img)
#
# copy1 = img
# copy2 = img2
#
# size = img.shape
#
# height = img.shape[0]
# width = img.shape[1]
# RGB = img.shape[2]
#
# mu, sigma = 0, 0.4  # mean and standard deviation
#
# # s = np.random.randint(0, 255, [3,3])
#
#
# RGB_ew = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_eb = np.random.normal(mu,sigma,[height, width, 1])
#
# w = np.random.normal(mu,sigma,[height, width, RGB])
# b = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_w1 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b1 = np.random.normal(mu,sigma,[height//4, width//4, RGB])
#
# ae_w2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_ww2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_bb2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ww1 = np.random.normal(mu,sigma,[height, width, RGB])
# bb1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# ww2 = np.random.normal(mu,sigma,[height, width, RGB])
# bb2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# RGB_dw1 = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_db1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# RGB_dw2 = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_db2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# # w_b = w
# # b_b = b
# # ww_b = ww
# # bb_b = bb
# #
# # feature1 = np.zeros([height//2, width//2, RGB])
# # active1 = np.zeros([height//2, width//2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature1[y][x][h] += copy1[2*y+n][2*x+m][h] * w[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
# #
# # feature2 = np.zeros([height//4, width//4, RGB])
# # active2 = np.zeros([height//4, width//4, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature2[y][x][h] += active1[2*y+n][2*x+m][h] * ae_w1[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             active2[y][x][h] = relu(feature2[y][x][h] + ae_b1[y][x][h])
# #
# # feature3 = np.zeros([height // 2, width // 2, RGB])
# # active3 = np.zeros([height // 2, width // 2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature3[2*y+n][2*x+m][h] += active2[y][x][h] * ae_w2[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active3[y][x][h] = relu(feature3[y][x][h] + ae_b2[y][x][h])
# #
# #
# # feature4 = np.zeros([height, width, RGB])
# # active4 = np.zeros([height, width, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature4[2*y+n][2*x+m][h] += active3[y][x][h] * ww[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width):
# #         for y in range(height):
# #             active4[y][x][h] = relu(feature4[y][x][h] + bb[y][x][h])
# #
# # loss = np.sum((np.square(copy1 - active4)) / (width*height))
# #
# # print(np.sum(loss))
# #
# # plot.imshow(active4)
# # plot.show()
# #
# # before = active4
# #
# # before_loss = loss
#
# for epoch in range(500):
#     print("epoch == {}".format(epoch))
#     if epoch % 2 == 0:
#         copy = seq.augment_image(copy1)
#         ae_dw2 = ae_w2
#         ae_db2 = ae_b2
#         ww = ww1
#         bb = bb1
#         RGB_dw = RGB_dw1
#         RGB_db = RGB_db1
#     else:
#         copy = seq.augment_image(copy2)
#         ae_dw2 = ae_ww2
#         ae_db2 = ae_bb2
#         ww = ww2
#         bb = bb2
#         RGB_dw = RGB_dw2
#         RGB_db = RGB_db2
#
#     # plot.imshow(copy)
#     # plot.show()
#
#     rate = 0.0
#
#     rgb_feature1 = np.zeros([height, width, 1])
#     rgb_active1 = np.zeros([height, width, 1])
#
#     for x in range(width):
#         for y in range(height):
#             for h in range(RGB):
#                 rgb_feature1[y][x][0] += copy[y][x][h] * RGB_ew[y][x][h]
#
#     for x in range(width):
#         for y in range(height):
#             rgb_active1[y][x][0] = relu(rgb_feature1[y][x][0])
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += rgb_active1[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)
#
#     feature2 = np.zeros([height // 4, width // 4, RGB])
#     active2 = np.zeros([height // 4, width // 4, RGB])
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)
#
#     feature3 = np.zeros([height // 2, width // 2, RGB])
#     active3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature3[2 * y + n][2 * x + m][h] += active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)
#
#     feature4 = np.zeros([height, width, RGB])
#     active4 = np.zeros([height, width, RGB])
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width):
#             for y in range(height):
#                 active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)
#
#     rgb_feature2 =np.zeros([height, width, RGB])
#     rgb_active2 =np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 rgb_feature2[y][x][h] += active4[y][x][0] * RGB_dw[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 rgb_active2[y][x][h] = relu_dropout(rgb_feature2[y][x][h] + RGB_db[y][x][h], rate)
#
#     loss = np.sum((np.square(copy - rgb_active2)) / (width * height))
#
#     print(np.sum(loss))
#
#     dH = rgb_active2 - copy
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#
#     # plot.imshow(active1.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
#     # plot.imshow(active2.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
#     # plot.imshow(active3.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))
#     # plot.imshow(active4.astype('uint8'))
#     # plot.show()
#     plot.imshow(rgb_active2.astype('uint8'))
#     plot.show()
#     plot.imshow(rgb_active2)
#     plot.show()
#
#     plot.imsave("epoch_{}_LR_{}_DROPOUT_{}.jpg".format(epoch, LR, rate), rgb_active2.astype('uint8'))
#
#     dRGB_dDW = np.zeros([height, width, RGB])
#     dRGB_dDB = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dDW[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0]) * active4[y][x][0]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dDB[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0])
#
#     dH_decode_RGB = np.zeros([height, width, 1])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dH_decode_RGB[y][x][0] += dH[y][x][h] * relu_prime(rgb_feature2[y][x][h]) * RGB_dw[y][x][h]
#
#
#     dWW = np.zeros([height, width, RGB])
#     dBB = np.zeros([height, width, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dWW[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * active3[y][x][h]
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dBB[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h])
#
#     dHH = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]
#
#     dAE_W2 = np.zeros([height//2, width//2, RGB])
#     dAE_B2 = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * active2[y][x][h]
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h])
#
#     dHH2 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * ae_dw2[2*y+n][2*x+m][h]
#
#     dAE_W1 = np.zeros([height//2, width//2, RGB])
#     dAE_B1 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W1[2*y+n][2*x+m][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * active1[2*y+n][2*x+m][h]
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B1[y][x][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h])
#
#     dHH3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH3[2*y+n][2*x+m][h] += dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * ae_w1[2*y+n][2*x+m][h]
#
#     dW = np.zeros([height, width, RGB])
#     dB = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[2*y+n][2*x+m][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h]) * rgb_active1[2*y+n][2*x+m][h]
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dB[y][x][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h])
#
#     dRGB_dEW = np.zeros([height, width, RGB])
#     dRGB_dEB = np.zeros([height, width, 1])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dEW[y][x][h] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0]) * copy[y][x][h]
#
#     for h in range(1):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dEB[y][x][0] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0])
#
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ae_w1 = ae_w1 - LR * dAE_W1
#     ae_b1 = ae_b1 - LR * dAE_B1
#
#     ae_dw2 = ae_dw2 - LR * dAE_W2
#     ae_db2 = ae_db2 - LR * dAE_B2
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     RGB_ew = RGB_ew - LR * dRGB_dEW
#     RGB_eb = RGB_eb - LR * dRGB_dEB
#
#     RGB_dw = RGB_dw - LR * dRGB_dDW
#     RGB_db = RGB_db - LR * dRGB_dDB
#
#     # w = weight_clipping_11_3D(w)
#     # ww = weight_clipping_11_3D(ww)
#     w = weight_clipping_11_3D(w)
#     ww = weight_clipping_11_3D(ww)
#
#     ae_w1 = weight_clipping_11_3D(ae_w1)
#     ae_w2 = weight_clipping_11_3D(ae_dw2)
#
#     RGB_ew = weight_clipping_11_3D(RGB_ew)
#     RGB_dw = weight_clipping_11_3D(RGB_dw)
#
#     w = weights_restart_3D_mask(w, 2)
#     ww = weights_restart_3D_mask(ww,2)
#
#     ae_w1 = weights_restart_3D_mask(ae_w1,2)
#     ae_w2 = weights_restart_3D_mask(ae_dw2,2)
#
#     RGB_ew = weights_restart_3D_RGB(RGB_ew)
#     RGB_dw = weights_restart_3D_RGB(RGB_dw)
#
#     # b = bias_clipping_1616_3D(b)
#     # bb = bias_clipping_1616_3D(bb)
#     #
#     # ae_b1 = bias_clipping_1616_3D(ae_b1)
#     # ae_b2 = bias_clipping_1616_3D(ae_b2)
#
#     if epoch % 2 == 0:
#         ae_w2 = ae_dw2
#         ae_b2 = ae_db2
#         ww1 = ww
#         bb1 = bb
#
#         RGB_dw1 = RGB_dw
#         RGB_db1 = RGB_db
#     else:
#         ae_ww2 = ae_dw2
#         ae_bb2 = ae_db2
#         ww2 = ww
#         bb2 = bb
#
#         RGB_dw2 = RGB_dw
#         RGB_db2 = RGB_db
#
# after = active4
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# # print("Answer == \n{}".format(copy))
# # print("Before active Value = \n{}".format(before.astype('int')))
# # print("After active Value = \n{}".format(after.astype('int')))
# print("=======================================")
#
# plot.imshow(img)
# plot.show()
#
# plot.imshow(active4.astype('uint8'))
# plot.show()
#
# plot.imsave("Output.jpg", active4.astype('uint8'))
#
# # print("w = \n {} \nb = {} \n".format(w, b))
# # print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# # print("ww = \n {} \nbb = {} \n".format(ww, bb))
# # print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
# #
# # print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))





#
#
# LR = 1
#
# img = plot.imread("edit_AKR20160511151000033_02_i.jpg")
#
# img2 = plot.imread("edit_33b1383713b057fc6e3097cee5902d35.jpg")
#
# # plot.imshow(img)
# # plot.show()
#
# print(img)
#
# copy1 = img
# copy2 = img2
#
# size = img.shape
#
# height = img.shape[0]
# width = img.shape[1]
# RGB = img.shape[2]
#
# mu, sigma = 0, 1  # mean and standard deviation
#
# # s = np.random.randint(0, 255, [3,3])
#
#
# RGB_ew = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_eb = np.random.normal(mu,sigma,[height, width, 1])
#
# w = np.random.normal(mu,sigma,[height, width, RGB])
# b = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_w1 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b1 = np.random.normal(mu,sigma,[height//4, width//4, RGB])
#
# ae_w2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_ww2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_bb2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ww1 = np.random.normal(mu,sigma,[height, width, RGB])
# bb1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# ww2 = np.random.normal(mu,sigma,[height, width, RGB])
# bb2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# RGB_dw1 = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_db1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# RGB_dw2 = np.random.normal(mu,sigma,[height, width, RGB])
# RGB_db2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# # w_b = w
# # b_b = b
# # ww_b = ww
# # bb_b = bb
# #
# # feature1 = np.zeros([height//2, width//2, RGB])
# # active1 = np.zeros([height//2, width//2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature1[y][x][h] += copy1[2*y+n][2*x+m][h] * w[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
# #
# # feature2 = np.zeros([height//4, width//4, RGB])
# # active2 = np.zeros([height//4, width//4, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature2[y][x][h] += active1[2*y+n][2*x+m][h] * ae_w1[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             active2[y][x][h] = relu(feature2[y][x][h] + ae_b1[y][x][h])
# #
# # feature3 = np.zeros([height // 2, width // 2, RGB])
# # active3 = np.zeros([height // 2, width // 2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature3[2*y+n][2*x+m][h] += active2[y][x][h] * ae_w2[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active3[y][x][h] = relu(feature3[y][x][h] + ae_b2[y][x][h])
# #
# #
# # feature4 = np.zeros([height, width, RGB])
# # active4 = np.zeros([height, width, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature4[2*y+n][2*x+m][h] += active3[y][x][h] * ww[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width):
# #         for y in range(height):
# #             active4[y][x][h] = relu(feature4[y][x][h] + bb[y][x][h])
# #
# # loss = np.sum((np.square(copy1 - active4)) / (width*height))
# #
# # print(np.sum(loss))
# #
# # plot.imshow(active4)
# # plot.show()
# #
# # before = active4
# #
# # before_loss = loss
#
# for epoch in range(500):
#     print("epoch == {}".format(epoch))
#     if epoch % 2 == 0:
#         copy = seq.augment_image(copy1)
#         ae_dw2 = ae_w2
#         ae_db2 = ae_b2
#         ww = ww1
#         bb = bb1
#         RGB_dw = RGB_dw1
#         RGB_db = RGB_db1
#     else:
#         copy = seq.augment_image(copy2)
#         ae_dw2 = ae_ww2
#         ae_db2 = ae_bb2
#         ww = ww2
#         bb = bb2
#         RGB_dw = RGB_dw2
#         RGB_db = RGB_db2
#
#     # plot.imshow(copy)
#     # plot.show()
#
#     rate = 0.0
#
#     rgb_feature1 = np.zeros([height, width, 1])
#     rgb_active1 = np.zeros([height, width, 1])
#
#     for x in range(width):
#         for y in range(height):
#             for h in range(RGB):
#                 rgb_feature1[y][x][0] += copy[y][x][h] * RGB_ew[y][x][h]
#
#     for x in range(width):
#         for y in range(height):
#             rgb_active1[y][x][0] = relu(rgb_feature1[y][x][0])
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += rgb_active1[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)
#
#     feature2 = np.zeros([height // 4, width // 4, RGB])
#     active2 = np.zeros([height // 4, width // 4, RGB])
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)
#
#     feature3 = np.zeros([height // 2, width // 2, RGB])
#     active3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature3[2 * y + n][2 * x + m][h] += active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)
#
#     feature4 = np.zeros([height, width, RGB])
#     active4 = np.zeros([height, width, RGB])
#
#     for h in range(1):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(1):
#         for x in range(width):
#             for y in range(height):
#                 active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)
#
#     rgb_feature2 =np.zeros([height, width, RGB])
#     rgb_active2 =np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 rgb_feature2[y][x][h] += active4[y][x][0] * RGB_dw[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 rgb_active2[y][x][h] = relu_dropout(rgb_feature2[y][x][h] + RGB_db[y][x][h], rate)
#
#     loss = np.sum((np.square(copy - rgb_active2)) / (width * height))
#
#     print(np.sum(loss))
#
#     dH = rgb_active2 - copy
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#
#     # plot.imshow(active1.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
#     # plot.imshow(active2.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
#     # plot.imshow(active3.astype('uint8'))
#     # plot.show()
#     # # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))
#     # plot.imshow(active4.astype('uint8'))
#     # plot.show()
#     plot.imshow(rgb_active2.astype('uint8'))
#     plot.show()
#     plot.imsave("epoch_{}_LR_{}_DROPOUT_{}.jpg".format(epoch, LR, rate), rgb_active2.astype('uint8'))
#
#     dRGB_dDW = np.zeros([height, width, RGB])
#     dRGB_dDB = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dDW[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0]) * active4[y][x][0]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dDB[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0])
#
#     dH_decode_RGB = np.zeros([height, width, 1])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dH_decode_RGB[y][x][0] += dH[y][x][h] * relu_prime(rgb_feature2[y][x][h]) * RGB_dw[y][x][h]
#
#
#     dWW = np.zeros([height, width, RGB])
#     dBB = np.zeros([height, width, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dWW[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * active3[y][x][h]
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dBB[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h])
#
#     dHH = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]
#
#     dAE_W2 = np.zeros([height//2, width//2, RGB])
#     dAE_B2 = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * active2[y][x][h]
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h])
#
#     dHH2 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * ae_dw2[2*y+n][2*x+m][h]
#
#     dAE_W1 = np.zeros([height//2, width//2, RGB])
#     dAE_B1 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W1[2*y+n][2*x+m][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * active1[2*y+n][2*x+m][h]
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B1[y][x][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h])
#
#     dHH3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(1):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH3[2*y+n][2*x+m][h] += dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * ae_w1[2*y+n][2*x+m][h]
#
#     dW = np.zeros([height, width, RGB])
#     dB = np.zeros([height//2, width//2, RGB])
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[2*y+n][2*x+m][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h]) * rgb_active1[2*y+n][2*x+m][h]
#
#     for h in range(1):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dB[y][x][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h])
#
#     dRGB_dEW = np.zeros([height, width, RGB])
#     dRGB_dEB = np.zeros([height, width, 1])
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dEW[y][x][h] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0]) * copy[y][x][h]
#
#     for h in range(1):
#         for x in range(width):
#             for y in range(height):
#                 dRGB_dEB[y][x][0] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0])
#
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ae_w1 = ae_w1 - LR * dAE_W1
#     ae_b1 = ae_b1 - LR * dAE_B1
#
#     ae_dw2 = ae_dw2 - LR * dAE_W2
#     ae_db2 = ae_db2 - LR * dAE_B2
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     RGB_ew = RGB_ew - LR * dRGB_dEW
#     RGB_eb = RGB_eb - LR * dRGB_dEB
#
#     RGB_dw = RGB_dw - LR * dRGB_dDW
#     RGB_db = RGB_db - LR * dRGB_dDB
#
#     # w = weight_clipping_11_3D(w)
#     # ww = weight_clipping_11_3D(ww)
#     w = weight_clipping_11_3D(w)
#     ww = weight_clipping_11_3D(ww)
#
#     ae_w1 = weight_clipping_11_3D(ae_w1)
#     ae_w2 = weight_clipping_11_3D(ae_dw2)
#
#     RGB_ew = weight_clipping_11_3D(RGB_ew)
#     RGB_dw = weight_clipping_11_3D(RGB_dw)
#
#     # b = bias_clipping_1616_3D(b)
#     # bb = bias_clipping_1616_3D(bb)
#     #
#     # ae_b1 = bias_clipping_1616_3D(ae_b1)
#     # ae_b2 = bias_clipping_1616_3D(ae_b2)
#
#     if epoch % 2 == 0:
#         ae_w2 = ae_dw2
#         ae_b2 = ae_db2
#         ww1 = ww
#         bb1 = bb
#
#         RGB_dw1 = RGB_dw
#         RGB_db1 = RGB_db
#     else:
#         ae_ww2 = ae_dw2
#         ae_bb2 = ae_db2
#         ww2 = ww
#         bb2 = bb
#
#         RGB_dw2 = RGB_dw
#         RGB_db2 = RGB_db
#
# after = active4
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# # print("Answer == \n{}".format(copy))
# # print("Before active Value = \n{}".format(before.astype('int')))
# # print("After active Value = \n{}".format(after.astype('int')))
# print("=======================================")
#
# plot.imshow(img)
# plot.show()
#
# plot.imshow(active4.astype('uint8'))
# plot.show()
#
# plot.imsave("Output.jpg", active4.astype('uint8'))
#
# # print("w = \n {} \nb = {} \n".format(w, b))
# # print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# # print("ww = \n {} \nbb = {} \n".format(ww, bb))
# # print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
# #
# # print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))



#
# LR = 1
#
# img = plot.imread("edit_AKR20160511151000033_02_i.jpg")
#
# img2 = plot.imread("edit_33b1383713b057fc6e3097cee5902d35.jpg")
#
# # plot.imshow(img)
# # plot.show()
#
# print(img)
#
# copy1 = img
# copy2 = img2
#
# size = img.shape
#
# height = img.shape[0]
# width = img.shape[1]
# RGB = img.shape[2]
#
# mu, sigma = 0, 1  # mean and standard deviation
#
# # s = np.random.randint(0, 255, [3,3])
#
# w = np.random.normal(mu,sigma,[height, width, RGB])
# b = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_w1 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b1 = np.random.normal(mu,sigma,[height//4, width//4, RGB])
#
# ae_w2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_b2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ae_ww2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
# ae_bb2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
#
# ww1 = np.random.normal(mu,sigma,[height, width, RGB])
# bb1 = np.random.normal(mu,sigma,[height, width, RGB])
#
# ww2 = np.random.normal(mu,sigma,[height, width, RGB])
# bb2 = np.random.normal(mu,sigma,[height, width, RGB])
#
# # w_b = w
# # b_b = b
# # ww_b = ww
# # bb_b = bb
# #
# # feature1 = np.zeros([height//2, width//2, RGB])
# # active1 = np.zeros([height//2, width//2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature1[y][x][h] += copy1[2*y+n][2*x+m][h] * w[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active1[y][x][h] = relu(feature1[y][x][h] + b[y][x][h])
# #
# # feature2 = np.zeros([height//4, width//4, RGB])
# # active2 = np.zeros([height//4, width//4, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature2[y][x][h] += active1[2*y+n][2*x+m][h] * ae_w1[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             active2[y][x][h] = relu(feature2[y][x][h] + ae_b1[y][x][h])
# #
# # feature3 = np.zeros([height // 2, width // 2, RGB])
# # active3 = np.zeros([height // 2, width // 2, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//4):
# #         for y in range(height//4):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature3[2*y+n][2*x+m][h] += active2[y][x][h] * ae_w2[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             active3[y][x][h] = relu(feature3[y][x][h] + ae_b2[y][x][h])
# #
# #
# # feature4 = np.zeros([height, width, RGB])
# # active4 = np.zeros([height, width, RGB])
# #
# # for h in range(RGB):
# #     for x in range(width//2):
# #         for y in range(height//2):
# #             for m in range(2):
# #                 for n in range(2):
# #                     feature4[2*y+n][2*x+m][h] += active3[y][x][h] * ww[2*y+n][2*x+m][h]
# #
# # for h in range(RGB):
# #     for x in range(width):
# #         for y in range(height):
# #             active4[y][x][h] = relu(feature4[y][x][h] + bb[y][x][h])
# #
# # loss = np.sum((np.square(copy1 - active4)) / (width*height))
# #
# # print(np.sum(loss))
# #
# # plot.imshow(active4)
# # plot.show()
# #
# # before = active4
# #
# # before_loss = loss
#
# for epoch in range(500):
#     print("epoch == {}".format(epoch))
#     # if epoch % 5 == 0:
#     #     if epoch != 0:
#     #         for pp in range(2):
#     #             if pp == 0:
#     #                 copy = img2
#     #                 ae_dw2 = ae_w2
#     #                 ae_db2 = ae_b2
#     #                 ww = ww1
#     #                 bb = bb1
#     #             elif pp == 1:
#     #                 copy = img
#     #                 ae_dw2 = ae_ww2
#     #                 ae_db2 = ae_bb2
#     #                 ww = ww2
#     #                 bb = bb2
#     #
#     #             rate = 0.0
#     #
#     #             feature1 = np.zeros([height // 2, width // 2, RGB])
#     #             active1 = np.zeros([height // 2, width // 2, RGB])
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width // 2):
#     #                     for y in range(height // 2):
#     #                         for m in range(2):
#     #                             for n in range(2):
#     #                                 feature1[y][x][h] += copy[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width // 2):
#     #                     for y in range(height // 2):
#     #                         active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)
#     #
#     #             feature2 = np.zeros([height // 4, width // 4, RGB])
#     #             active2 = np.zeros([height // 4, width // 4, RGB])
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width // 4):
#     #                     for y in range(height // 4):
#     #                         for m in range(2):
#     #                             for n in range(2):
#     #                                 feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width // 4):
#     #                     for y in range(height // 4):
#     #                         active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)
#     #
#     #             feature3 = np.zeros([height // 2, width // 2, RGB])
#     #             active3 = np.zeros([height // 2, width // 2, RGB])
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width // 4):
#     #                     for y in range(height // 4):
#     #                         for m in range(2):
#     #                             for n in range(2):
#     #                                 feature3[2 * y + n][2 * x + m][h] += active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width // 2):
#     #                     for y in range(height // 2):
#     #                         active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)
#     #
#     #             feature4 = np.zeros([height, width, RGB])
#     #             active4 = np.zeros([height, width, RGB])
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width // 2):
#     #                     for y in range(height // 2):
#     #                         for m in range(2):
#     #                             for n in range(2):
#     #                                 feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]
#     #
#     #             for h in range(RGB):
#     #                 for x in range(width):
#     #                     for y in range(height):
#     #                         active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)
#     #
#     #             loss = np.sum((np.square(copy - active4)) / (width * height))
#     #
#     #             # plot.imshow(active1.astype('uint8'))
#     #             # plot.show()
#     #             # # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
#     #             # plot.imshow(active2.astype('uint8'))
#     #             # plot.show()
#     #             # # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
#     #             # plot.imshow(active3.astype('uint8'))
#     #             # plot.show()
#     #             # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))
#     #
#     #             plot.imshow(active4.astype('uint8'))
#     #             plot.show()
#     #             plot.imsave("aug_Convert_input__epoch_{}_{}.jpg".format(pp, epoch), active4.astype('uint8'))
#     if epoch % 2 == 0:
#         copy = copy1
#         ae_dw2 = ae_w2
#         ae_db2 = ae_b2
#         ww = ww1
#         bb = bb1
#     else:
#         copy = copy2
#         ae_dw2 = ae_ww2
#         ae_db2 = ae_bb2
#         ww = ww2
#         bb = bb2
#
#     rate = 0.0
#
#     feature1 = np.zeros([height // 2, width // 2, RGB])
#     active1 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature1[y][x][h] += copy[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)
#
#     feature2 = np.zeros([height // 4, width // 4, RGB])
#     active2 = np.zeros([height // 4, width // 4, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)
#
#     feature3 = np.zeros([height // 2, width // 2, RGB])
#     active3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 4):
#             for y in range(height // 4):
#                 for m in range(2):
#                     for n in range(2):
#                         feature3[2 * y + n][2 * x + m][h] += active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)
#
#     feature4 = np.zeros([height, width, RGB])
#     active4 = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width // 2):
#             for y in range(height // 2):
#                 for m in range(2):
#                     for n in range(2):
#                         feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]
#
#     for h in range(RGB):
#         for x in range(width):
#             for y in range(height):
#                 active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)
#
#     loss = np.sum((np.square(copy - active4)) / (width * height))
#
#     print(np.sum(loss))
#
#     dH = active4 - copy
#
#     # plot.imshow(dH.astype('uint8'))
#     # plot.show()
#
#     # dH = seq2.augment_image(dH)
#
#     #
#     # max_active = active2 / active2.max()
#     #
#     # show_img = active2 / max_active * 255
#     #
#     # show_img = show_img.astype('uint8')
#
#     # plot.imshow(active1.astype('uint8'))
#     # plot.show()
#     # # # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
#     # plot.imshow(active2.astype('uint8'))
#     # plot.show()
#     # # # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
#     # plot.imshow(active3.astype('uint8'))
#     # plot.show()
#     # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))
#     if epoch %2 == 0:
#         plot.imshow(active4.astype('uint8'))
#         plot.show()
#         plot.imsave("30per_dHH2/epoch_{}_LR_{}_weightrestart_aver_1over_random30percent_loss_{}.jpg".format(epoch, LR, int(loss)), active4.astype('uint8'))
#
#     dWW = np.zeros([height, width, RGB])
#     dBB = np.zeros([height, width, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dWW[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * active3[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dBB[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h])
#
#     dHH = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]
#
#     dAE_W2 = np.zeros([height//2, width//2, RGB])
#     dAE_B2 = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * active2[y][x][h]
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h])
#
#     dHH2 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH2[y][x][h] += dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * ae_dw2[2*y+n][2*x+m][h]
#
#     dAE_W1 = np.zeros([height//2, width//2, RGB])
#     dAE_B1 = np.zeros([height//4, width//4, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_W1[2*y+n][2*x+m][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * active1[2*y+n][2*x+m][h]
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dAE_B1[y][x][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h])
#
#     dHH3 = np.zeros([height // 2, width // 2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//4):
#             for y in range(height//4):
#                 for m in range(2):
#                     for n in range(2):
#                         dHH3[2*y+n][2*x+m][h] += dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * ae_w1[2*y+n][2*x+m][h]
#
#     dW = np.zeros([height, width, RGB])
#     dB = np.zeros([height//2, width//2, RGB])
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dW[2*y+n][2*x+m][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h]) * copy[2*y+n][2*x+m][h]
#
#     for h in range(RGB):
#         for x in range(width//2):
#             for y in range(height//2):
#                 for m in range(2):
#                     for n in range(2):
#                         dB[y][x][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h])
#
#     w = w - LR * dW
#     b = b - LR * dB
#
#     ae_w1 = ae_w1 - LR * dAE_W1
#     ae_b1 = ae_b1 - LR * dAE_B1
#
#     ae_dw2 = ae_dw2 - LR * dAE_W2
#     ae_db2 = ae_db2 - LR * dAE_B2
#
#     ww = ww - LR * dWW
#     bb = bb - LR * dBB
#
#     # w = weight_clipping_11_3D(w)
#     # ww = weight_clipping_11_3D(ww)
#     w = weight_clipping_11_3D(w)
#     ww = weight_clipping_11_3D(ww)
#
#     ae_w1 = weight_clipping_11_3D(ae_w1)
#     ae_dw2 = weight_clipping_11_3D(ae_dw2)
#
#
#     # w = weights_restart_3D_mask(w,2, b)
#     # ww = weights_restart_3D_mask(ww,2 , bb)
#     #
#     # ae_w1 = weights_restart_3D_mask(ae_w1,2, ae_b1)
#     # ae_dw2 = weights_restart_3D_mask(ae_dw2,2, ae_db2)
#
#     w = weights_average_filter_3D_mask(w,2, b)
#     ww = weights_average_filter_3D_mask(ww,2 , bb)
#
#     ae_w1 = weights_average_filter_3D_mask(ae_w1,2, ae_b1)
#     ae_dw2 = weights_average_filter_3D_mask(ae_dw2,2, ae_db2)
#
#     # b = bias_clipping_128128_3D(b)
#     # bb = bias_clipping_128128_3D(bb)
#     #
#     # ae_b1 = bias_clipping_128128_3D(ae_b1)
#     # ae_b2 = bias_clipping_128128_3D(ae_b2)
#
#     if epoch % 2 == 0:
#         ae_w2 = ae_dw2
#         ae_b2 = ae_db2
#         ww1 = ww
#         bb1 = bb
#     else:
#         ae_ww2 = ae_dw2
#         ae_bb2 = ae_db2
#         ww2 = ww
#         bb2 = bb
#
#
# after = active4
#
# after_loss = np.sum(loss)
#
# print("\n\n")
# # print("Answer == \n{}".format(copy))
# # print("Before active Value = \n{}".format(before.astype('int')))
# # print("After active Value = \n{}".format(after.astype('int')))
# print("=======================================")
#
# plot.imshow(img)
# plot.show()
#
# plot.imshow(active4.astype('uint8'))
# plot.show()
#
# plot.imsave("Output.jpg", active4.astype('uint8'))
#
# # print("w = \n {} \nb = {} \n".format(w, b))
# # print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# # print("ww = \n {} \nbb = {} \n".format(ww, bb))
# # print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
# print("=======================================")
# #
# # print("Before LOSS = {}".format(before_loss))
# print("After LOSS = {}".format(after_loss))




LR = 1

img = plot.imread("edit_AKR20160511151000033_02_i.jpg")

img2 = plot.imread("edit_33b1383713b057fc6e3097cee5902d35.jpg")

# plot.imshow(img)
# plot.show()

print(img)

copy1 = img
copy2 = img2

size = img.shape

height = img.shape[0]
width = img.shape[1]
RGB = img.shape[2]

mu, sigma = 0, 1  # mean and standard deviation

fc_out1 = 2

# s = np.random.randint(0, 255, [3,3])

w = np.random.normal(mu,sigma,[height, width, RGB])
b = np.random.normal(mu,sigma,[height//2, width//2, RGB])

ae_w1 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
ae_b1 = np.random.normal(mu,sigma,[height//4, width//4, RGB])

fc_ew1 = np.random.normal(mu, sigma, [height//4, width//4, RGB, fc_out1])
fc_eb1 = np.random.normal(mu, sigma, [fc_out1])

fc_dw1 = np.random.normal(mu, sigma, [height//4, width//4, RGB, fc_out1])
fc_db1 = np.random.normal(mu, sigma, [height//4, width//4, RGB, fc_out1])

fc_dw2 = np.random.normal(mu, sigma, [height//4, width//4, RGB, fc_out1])
fc_db2 = np.random.normal(mu, sigma, [height//4, width//4, RGB, fc_out1])

ae_w2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
ae_b2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])

ae_ww2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
ae_bb2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])

ww1 = np.random.normal(mu,sigma,[height, width, RGB])
bb1 = np.random.normal(mu,sigma,[height, width, RGB])

ww2 = np.random.normal(mu,sigma,[height, width, RGB])
bb2 = np.random.normal(mu,sigma,[height, width, RGB])

for epoch in range(500):
    print("epoch == {}".format(epoch))
    # if epoch % 5 == 0:
    #     if epoch != 0:
    #         for pp in range(2):
    #             if pp == 0:
    #                 copy = img2
    #                 ae_dw2 = ae_w2
    #                 ae_db2 = ae_b2
    #                 ww = ww1
    #                 bb = bb1
    #             elif pp == 1:
    #                 copy = img
    #                 ae_dw2 = ae_ww2
    #                 ae_db2 = ae_bb2
    #                 ww = ww2
    #                 bb = bb2

    if epoch % 2 == 0:
        copy = copy1
        ae_dw2 = ae_w2
        ae_db2 = ae_b2
        ww = ww1
        bb = bb1
        fc_dw = fc_dw1
        fc_db = fc_db1
    else:
        copy = copy2
        ae_dw2 = ae_ww2
        ae_db2 = ae_bb2
        ww = ww2
        bb = bb2
        fc_dw = fc_dw2
        fc_db = fc_db2

    rate = 0.0
    fc_dropout_rate = 0.0

    feature1 = np.zeros([height // 2, width // 2, RGB])
    active1 = np.zeros([height // 2, width // 2, RGB])

    for h in range(RGB):
        for x in range(width // 2):
            for y in range(height // 2):
                for m in range(2):
                    for n in range(2):
                        feature1[y][x][h] += copy[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]

    for h in range(RGB):
        for x in range(width // 2):
            for y in range(height // 2):
                active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)

    feature2 = np.zeros([height // 4, width // 4, RGB])
    active2 = np.zeros([height // 4, width // 4, RGB])

    for h in range(RGB):
        for x in range(width // 4):
            for y in range(height // 4):
                for m in range(2):
                    for n in range(2):
                        feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]

    for h in range(RGB):
        for x in range(width // 4):
            for y in range(height // 4):
                active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)

    full_conn_feature1 = np.zeros([fc_out1])
    full_conn_active1 = np.zeros([fc_out1])

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width // 4):
                for y in range(height // 4):
                    full_conn_feature1[fc] += active2[y][x][h] * fc_ew1[y][x][h][fc]

    for fc in range(fc_out1):
        full_conn_active1[fc] = relu_dropout(full_conn_feature1[fc] + fc_eb1[fc], fc_dropout_rate)

    full_conn_feature2 = np.zeros([height // 4, width // 4, RGB])
    full_conn_active2 = np.zeros([height // 4, width // 4, RGB])

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width // 4):
                for y in range( height // 4):
                    full_conn_feature2[y][x][h] += full_conn_active1[fc] * fc_dw[y][x][h][fc]

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width // 4):
                for y in range(height // 4):
                    full_conn_active2[y][x][h] = relu_dropout(full_conn_feature2[y][x][h] + fc_db[y][x][h][fc], fc_dropout_rate)

    feature3 = np.zeros([height // 2, width // 2, RGB])
    active3 = np.zeros([height // 2, width // 2, RGB])

    for h in range(RGB):
        for x in range(width // 4):
            for y in range(height // 4):
                for m in range(2):
                    for n in range(2):
                        feature3[2 * y + n][2 * x + m][h] += full_conn_active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]

    for h in range(RGB):
        for x in range(width // 2):
            for y in range(height // 2):
                active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)

    feature4 = np.zeros([height, width, RGB])
    active4 = np.zeros([height, width, RGB])

    for h in range(RGB):
        for x in range(width // 2):
            for y in range(height // 2):
                for m in range(2):
                    for n in range(2):
                        feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]

    for h in range(RGB):
        for x in range(width):
            for y in range(height):
                active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)

    loss = np.sum((np.square(copy - active4)) / (width * height))

    print(np.sum(loss))

    dH = active4 - copy

    # plot.imshow(dH.astype('uint8'))
    # plot.show()

    # dH = seq2.augment_image(dH)

    #
    # max_active = active2 / active2.max()
    #
    # show_img = active2 / max_active * 255
    #
    # show_img = show_img.astype('uint8')

    # plot.imshow(active1.astype('uint8'))
    # plot.show()
    # # # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
    # plot.imshow(active2.astype('uint8'))
    # plot.show()
    # # # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
    # plot.imshow(active3.astype('uint8'))
    # plot.show()
    # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))

    if epoch %2 == 0:
        plot.imshow(active4.astype('uint8'))
        plot.show()
        plot.imsave("FCN/epoch_{}_LR_{}_weightrestart_aver_1over_random100percent_loss_{}.jpg".format(epoch, LR, int(loss)), active4.astype('uint8'))

    dWW = np.zeros([height, width, RGB])
    dBB = np.zeros([height, width, RGB])

    for h in range(RGB):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dWW[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * active3[y][x][h]

    for h in range(RGB):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dBB[2*y+n][2*x+m][h] = dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h])

    dHH = np.zeros([height//2, width//2, RGB])

    for h in range(RGB):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]

    dAE_W2 = np.zeros([height//2, width//2, RGB])
    dAE_B2 = np.zeros([height//2, width//2, RGB])

    for h in range(RGB):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_W2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * full_conn_active2[y][x][h]

    for h in range(RGB):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_B2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h])

    dHH2 = np.zeros([height//4, width//4, RGB])

    for h in range(RGB):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dHH2[y][x][h] += dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * ae_dw2[2*y+n][2*x+m][h]

    dFC_W2 = np.zeros([height // 4, width // 4, RGB, fc_out1])
    dFC_B2 = np.zeros([height // 4, width // 4, RGB, fc_out1])

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width//4):
                for y in range(height//4):
                    dFC_W2[y][x][h][fc] += dHH2[y][x][h] * relu_prime(full_conn_feature2[y][x][h]) * full_conn_active1[fc]

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width//4):
                for y in range(height//4):
                    dFC_B2[y][x][h][fc] = dHH2[y][x][h] * relu_prime(full_conn_feature2[y][x][h])

    dHH_FC2 = np.zeros([fc_out1])

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width//4):
                for y in range(height//4):
                    dHH_FC2[fc] += dHH2[y][x][h] * relu_prime(full_conn_feature2[y][x][h]) * fc_dw[y][x][h][fc]

    dFC_W1 = np.zeros([height // 4, width // 4, RGB, fc_out1])
    dFC_B1 = np.zeros([fc_out1])

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width//4):
                for y in range(height//4):
                    dFC_W1[y][x][h][fc] = dHH_FC2[fc] * relu_prime(full_conn_feature1[fc]) * active2[y][x][h]


    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width//4):
                for y in range(height//4):
                    dFC_B1[fc] += dHH_FC2[fc] * relu_prime(full_conn_feature1[fc])

    dHH_FC1 = np.zeros([height//4, width//4, RGB])

    for fc in range(fc_out1):
        for h in range(RGB):
            for x in range(width//4):
                for y in range(height//4):
                    dHH_FC1[y][x][h] = dHH_FC2[fc] * relu_prime(full_conn_feature1[fc]) * fc_ew1[y][x][h][fc]

    dAE_W1 = np.zeros([height//2, width//2, RGB])
    dAE_B1 = np.zeros([height//4, width//4, RGB])

    for h in range(RGB):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_W1[2*y+n][2*x+m][h] = dHH_FC1[y][x][h] * relu_prime(feature2[y][x][h]) * active1[2*y+n][2*x+m][h]

    for h in range(RGB):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_B1[y][x][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h])

    dHH3 = np.zeros([height // 2, width // 2, RGB])

    for h in range(RGB):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dHH3[2*y+n][2*x+m][h] += dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * ae_w1[2*y+n][2*x+m][h]

    dW = np.zeros([height, width, RGB])
    dB = np.zeros([height//2, width//2, RGB])

    for h in range(RGB):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dW[2*y+n][2*x+m][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h]) * copy[2*y+n][2*x+m][h]

    for h in range(RGB):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dB[y][x][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h])

    w = w - LR * dW
    b = b - LR * dB

    ae_w1 = ae_w1 - LR * dAE_W1
    ae_b1 = ae_b1 - LR * dAE_B1

    fc_ew1 = fc_ew1 - LR * dFC_W1
    fc_eb1 = fc_eb1 - LR * dFC_B1

    fc_dw = fc_dw - LR * dFC_W2
    fc_db = fc_db - LR * dFC_B2

    ae_dw2 = ae_dw2 - LR * dAE_W2
    ae_db2 = ae_db2 - LR * dAE_B2

    ww = ww - LR * dWW
    bb = bb - LR * dBB

    w = weight_clipping_11_3D(w)
    ww = weight_clipping_11_3D(ww)

    ae_w1 = weight_clipping_11_3D(ae_w1)
    ae_dw2 = weight_clipping_11_3D(ae_dw2)


    w = weights_restart_3D_mask(w,2, b)
    ww = weights_restart_3D_mask(ww,2 , bb)

    ae_w1 = weights_restart_3D_mask(ae_w1,2, ae_b1)
    ae_dw2 = weights_restart_3D_mask(ae_dw2,2, ae_db2)

    fc_ew1 = weights_restart_4D_mask(fc_ew1, 1, fc_eb1)
    fc_dw = weights_restart_4D_mask(fc_dw1, 1, fc_db)

    # w = weights_average_filter_3D_mask(w,2, b)
    # ww = weights_average_filter_3D_mask(ww,2 , bb)
    #
    # ae_w1 = weights_average_filter_3D_mask(ae_w1,2, ae_b1)
    # ae_dw2 = weights_average_filter_3D_mask(ae_dw2,2, ae_db2)

    # b = bias_clipping_128128_3D(b)
    # bb = bias_clipping_128128_3D(bb)
    #
    # ae_b1 = bias_clipping_128128_3D(ae_b1)
    # ae_b2 = bias_clipping_128128_3D(ae_b2)

    if epoch % 2 == 0:
        ae_w2 = ae_dw2
        ae_b2 = ae_db2
        ww1 = ww
        bb1 = bb
        fc_dw1 = fc_dw
        fc_db1 = fc_db
    else:
        ae_ww2 = ae_dw2
        ae_bb2 = ae_db2
        ww2 = ww
        bb2 = bb
        fc_dw2 = fc_dw
        fc_db2 = fc_db


after = active4

after_loss = np.sum(loss)

print("\n\n")
# print("Answer == \n{}".format(copy))
# print("Before active Value = \n{}".format(before.astype('int')))
# print("After active Value = \n{}".format(after.astype('int')))
print("=======================================")

plot.imshow(img)
plot.show()

plot.imshow(active4.astype('uint8'))
plot.show()

plot.imsave("Output.jpg", active4.astype('uint8'))

# print("w = \n {} \nb = {} \n".format(w, b))
# print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
# print("ww = \n {} \nbb = {} \n".format(ww, bb))
# print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
print("=======================================")
#
# print("Before LOSS = {}".format(before_loss))
print("After LOSS = {}".format(after_loss))


