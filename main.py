import numpy as np
import random
import matplotlib
import PIL
import matplotlib.pyplot as plot

def bias_clipping_1616_1D(input_arr):
    x = input_arr
    for i in range(x.shape[0]):
        if x[i] > 15:
            x[i] = 15
        elif x[i] < -16:
            x[i] = -16
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
        return x / 20


def relu_prime(x):
    if x >= 0:
        return 1
    else:
        return 1 / 20


LR = 0.0000001


mu, sigma = 0, 1  # mean and standard deviation

# s == input (3x3)
# w == encoder weight (2x2)
# ww == decoder weight (2x2)
# b, bb == biases
s = np.random.randint(0, 255, [3,3])
w = np.random.normal(mu,sigma,[2,2])
b = np.random.normal(mu,sigma,[2,2])

aw1 = np.random.normal(mu, sigma, [2,2,2])
ab1 = np.random.normal(mu, sigma, [2,2])

aw2 = np.random.normal(mu, sigma, [2,2,2])
ab2 = np.random.normal(mu, sigma, [2,2])


ww = np.random.normal(mu,sigma,[2,2])
bb = np.random.normal(mu,sigma,[3,3])

# temporary initial forward values
w_b = w
b_b = b
ww_b = ww
bb_b = bb

# feature1 == feature map from encoder weight
# active1 == activation map from encoder weight

feature1 = np.zeros([2,2])
active1 = np.zeros([2,2])

for x in range(2):
    for y in range(2):
        for m in range(2):
            for n in range(2):
                feature1[x][y] += w[m][n] * s[y+m][x+n]

for x in range(2):
    for y in range(2):
        active1[x][y] = relu(feature1[x][y] + b[x][y])

feature2 = np.zeros([3, 3])
active2 = np.zeros([3, 3])

# reshape for Transposed convolution
# not essential but for my convenience
active1_reshape = active1.reshape([4])

# Transpose plane
# For more information,
# https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/
# checkout for Transposed Convolution matrix

# In the link, He constructed Transposed convolution matrix with (out width * out height, (mask size^2))
# but I've constructed Transposed convolution matrix with (out width, out height, mask size^2)
TransposePlane = np.zeros([4,3,3])

for x in range(2):
    for y in range(2):
        TransposePlane[0][x][y] = ww[x][y]
        TransposePlane[1][x][y +1] = ww[x][y]
        TransposePlane[2][x+1][y] = ww[x][y]
        TransposePlane[3][x + 1][y + 1] = ww[x][y]

for m in range(3):
    for n in range(3):
        for h in range(4):
            feature2[m][n] += active1_reshape[h] * TransposePlane[h][m][n]



for m in range(3):
    for n in range(3):
        active2[m][n] = relu(feature2[m][n] + bb[m][n])

loss = np.zeros([3, 3])

# simple square mean loss
loss = (np.square(s - active2)) / 9

print(np.sum(loss))

before = active2

before_loss = np.sum((np.square(s - active2)) / 2)

for i in range(10000):

    feature1 = np.zeros([2, 2])
    active1 = np.zeros([2, 2])

    for x in range(2):
        for y in range(2):
            for m in range(2):
                for n in range(2):
                    feature1[x][y] += w[m][n] * s[y + m][x + n]

    for x in range(2):
        for y in range(2):
            active1[x][y] = relu(feature1[x][y] + b[x][y])

    feature2 = np.zeros([3, 3])
    active2 = np.zeros([3, 3])

    active1_reshape = active1.reshape([4])

    # Transpose plane
    TransposePlane = np.zeros([4, 3, 3])

    for x in range(2):
        for y in range(2):
            TransposePlane[0][x][y] = ww[x][y]
            TransposePlane[1][x][y + 1] = ww[x][y]
            TransposePlane[2][x + 1][y] = ww[x][y]
            TransposePlane[3][x + 1][y + 1] = ww[x][y]

    for m in range(3):
        for n in range(3):
            for h in range(4):
                feature2[m][n] += active1_reshape[h] * TransposePlane[h][m][n]

    for m in range(3):
        for n in range(3):
            active2[m][n] = relu(feature2[m][n] + bb[m][n])

    loss = (np.square(s - active2)) / 9

    print(np.sum(loss))

    dH = active2 - s

    dWW = np.zeros([2,2])
    dBB = np.zeros([3,3])

    for x in range(2):
        for y in range(2):
            for m in range(2):
                for n in range(2):
                    dWW[x][y] += dH[x+m][y+n] * relu_prime(feature2[x+m][y+n]) * active1[m][n]

    for x in range(3):
        for y in range(3):
            dBB[x][y] += dH[x][y] * relu_prime(feature2[x][y])

    dHH = np.zeros([2,2])

    for x in range(2):
        for y in range(2):
            for m in range(2):
                for n in range(2):
                    dHH[x][y] += dH[x+m][y+n] * relu_prime(feature2[x+m][y+n]) * ww[m][n]

    dW = np.zeros([2,2])
    dB = np.zeros([2,2])

    for x in range(2):
        for y in range(2):
            for m in range(2):
                for n in range(2):
                    dW[x][y] += dHH[x][y] * relu_prime(feature1[x][y]) * s[x+m][y+n]

    for x in range(2):
        for y in range(2):
            dB[x][y] += dHH[x][y] * relu_prime(feature1[x][y])

    w = w - LR * dW
    b = b - LR * dB

    ww = ww - LR * dWW
    bb = bb - LR * dBB

    # w = weight_clipping_1616_2D(w)
    # ww = weight_clipping_1616_2D(ww)

    # b = bias_clipping_1616_1D(b)
    # bb = bias_clipping_1616_1D(bb)


after = active2

after_loss = np.sum(loss)

print("\n\n")
print("Answer == \n{}".format(s))
print("Before active Value = \n{}".format(before))
print("After active Value = \n{}".format(after))
print("=======================================")

print("w = \n {} \nb = {} \n".format(w, b))
print("w_b = \n {} \nb_b = {} \n\n".format(w_b, b_b))
print("ww = \n {} \nbb = {} \n".format(ww, bb))
print("ww_b = \n {} \nbb_b = {} \n".format(ww_b, bb_b))
print("=======================================")

print("Before LOSS = {}".format(before_loss))
print("After LOSS = {}".format(after_loss))

