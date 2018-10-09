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
    for h in range(x.shape[2]):
        for i in range(x.shape[1]//mask_size):
            for j in range(x.shape[0]//mask_size):
                count = 0
                for m in range(mask_size):
                    for n in range(mask_size):
                        if x[mask_size*j + n][mask_size*i+m][h] == -1 \
                                or x[mask_size*j + n][mask_size*i+m][h] == 1:
                            count +=1
                # if count >= (mask_size**2)-2:
                if count >= 1:
                    for m in range(mask_size):
                        for n in range(mask_size):
                            x[mask_size * j + n][mask_size * i + m][h] = random.random() * 2 - 1
                            try:
                                bias[mask_size * j + n][mask_size * i + m][h] = 0
                            except:
                                pass
                            try:
                                bias[j][i][h] = 0
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


LR = 0.1


s1 = np.random.randint(0, 255, [3,3])
s2 = np.random.randint(0, 255, [3,3])

w = np.random.randint(-1,1,[2,3,3])

b = np.random.randint(-1,1,[2])

ww1 = np.random.randint(-1,1,[2,3,3])

bb1 = np.random.randint(-1,1,[3,3])

ww2 = np.random.randint(-1,1,[2,3,3])

bb2 = np.random.randint(-1,1,[3,3])

w_b = w
b_b = b
ww_b1 = ww1
bb_b1 = bb1
ww_b2 = ww2
bb_b2 = bb2


for pp in range(2):
    if pp == 0 :
        ww = ww1
        bb = bb1
        s = s1
    elif pp == 1 :
        ww = ww2
        bb = bb2
        s = s2
    feature1 = np.zeros([2])
    active1 = np.zeros([2])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                feature1[h] += s[m][n] * w[h][m][n]

    for h in range(2):
        active1[h] = relu(feature1[h] + b[h])

    feature2 = np.zeros([3, 3])
    active2 = np.zeros([3, 3])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                feature2[m][n] += active1[h] * ww[h][m][n]

    for m in range(3):
        for n in range(3):
            active2[m][n] = relu(feature2[m][n] + bb[m][n])

    loss = np.zeros([3, 3])

    loss = (np.square(s - active2)) / 2

    print(np.sum(loss))

    before = active2

    before_loss = np.sum((np.square(s - active2)) / 2)

for i in range(100):

    if i % 2 == 0 :
        ww = ww1
        bb = bb1
        s = s1
    elif i % 2 == 1 :
        ww = ww2
        bb = bb2
        s = s2

    feature1 = np.zeros([2])
    active1 = np.zeros([2])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                feature1[h] += s[m][n] * w[h][m][n]

    for h in range(2):
        active1[h] = relu(feature1[h] + b[h])

    feature2 = np.zeros([3, 3])
    active2 = np.zeros([3, 3])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                feature2[m][n] += active1[h] * ww[h][m][n]

    for m in range(3):
        for n in range(3):
            active2[m][n] = relu(feature2[m][n] + bb[m][n])

    loss = np.zeros([3,3])

    # for m in range(3):
    #     for n in range(3):
    #         dH[m][n] = -(s[m][n] * np.log10(active2[m][n]) + (1 - s[m][n]) * np.log10(1 - active2[m][n]))

    loss = (np.square(s - active2)) / 2

    print(np.sum(loss))

    dH = active2 - s

    dWW = np.zeros([2,3,3])
    dBB = np.zeros([3,3])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                dWW[h][m][n] = dH[m][n] * relu_prime(feature2[m][n]) * active1[h]

    for m in range(3):
        for n in range(3):
            dBB[m][n] = dH[m][n] * relu_prime(feature2[m][n])

    dHH = np.zeros([2])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                dHH[h] += dH[m][n] * relu_prime(feature2[m][n]) * ww[h][m][n]

    dW = np.zeros([2,3,3])
    dB = np.zeros([2])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                dW[h][m][n] = dHH[h] * relu_prime(feature1[h]) * s[m][n]

    for h in range(2):
        dB[h] = dHH[h] * relu_prime(feature1[h])

    w = w - LR * dW
    b = b - LR * dB

    ww = ww - LR * dWW
    bb = bb - LR * dBB

    # w = weight_clipping_11_3D(w)
    # ww = weight_clipping_11_3D(ww)

    if i % 2 == 0 :
        ww1 = ww
        bb1 = bb
    elif i % 2 == 1 :
        ww2 = ww
        bb2 = bb


for i in range(4):

    if i == 0 :
        ww = ww1
        bb = bb1
        s = s1
    elif i == 1 :
        ww = ww2
        bb = bb2
        s = s2
    elif i == 2 :
        ww = ww2
        bb = bb2
        s = s1
    elif i == 3 :
        ww = ww1
        bb = bb1
        s = s2

    feature1 = np.zeros([2])
    active1 = np.zeros([2])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                feature1[h] += s[m][n] * w[h][m][n]

    for h in range(2):
        active1[h] = relu(feature1[h] + b[h])

    feature2 = np.zeros([3, 3])
    active2 = np.zeros([3, 3])

    for h in range(2):
        for m in range(3):
            for n in range(3):
                feature2[m][n] += active1[h] * ww[h][m][n]

    for m in range(3):
        for n in range(3):
            active2[m][n] = relu(feature2[m][n] + bb[m][n])

    loss = np.zeros([3,3])

    # for m in range(3):
    #     for n in range(3):
    #         dH[m][n] = -(s[m][n] * np.log10(active2[m][n]) + (1 - s[m][n]) * np.log10(1 - active2[m][n]))

    loss = (np.square(s - active2)) / 2

    print(np.sum(loss))
    print(s)
    print(active2)



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

