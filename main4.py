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

def weights_average_filter_3D_RGB(input_arr):
    x = input_arr
    for m in range(x.shape[1]):
        for n in range(x.shape[0]):
            count = 0
            for h in range(x.shape[2]):
                if x[n][m][h] == -1 or x[n][m][h] == 1:
                    count += 1
            if count > 1:
                for h in range(x.shape[2]):
                    x[n][m][h] = random.random() * 2 - 1

    return x
    x = input_arr

    for i in range(x.shape[1] // mask_size):
        for j in range(x.shape[0] // mask_size):
            count = 0
            for h in range(x.shape[2]):
                for m in range(mask_size):
                    for n in range(mask_size):
                        if x[mask_size * j + n][mask_size * i + m][h] == -1 \
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
                                temp += x[mask_size * j + n][mask_size * i + m - space ][h]
                                edge += 1
                            except:
                                pass

                            x[mask_size * j + n][mask_size * i + m][h] = (1 * random.random() /3 + ((2*temp) /(3* edge)))
                            try:
                                bias[mask_size * j + n][mask_size * i + m][h] = 0
                            except:
                                pass
                            try:
                                bias[j][i][h] = 0
                            except:
                                pass

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

                            x[mask_size * j + n][mask_size * i + m][h] = (1 * random.random() /3 + ((2*temp) /(3* edge)))
                            try:
                                bias[mask_size * j + n][mask_size * i + m][h] = 0
                            except:
                                pass
                            try:
                                bias[j][i][h] = 0
                            except:
                                pass

    return x

def weights_average_filter_3D_mask_101(input_arr, mask_size , *bias):
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

                            x[mask_size * j + n][mask_size * i + m][h] = (1 * random.random() /3 + ((2*temp) /(3* edge)))
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




LR = 1

img = plot.imread("edit_AKR20160511151000033_02_i.jpg")

img2 = plot.imread("edit_33b1383713b057fc6e3097cee5902d35.jpg")

# plot.imshow(img)
# plot.show()

# print(img)

copy1 = img
copy2 = img2

size = img.shape

height = img.shape[0]
width = img.shape[1]
RGB = img.shape[2]

mu, sigma = 0, 0.4  # mean and standard deviation

# s = np.random.randint(0, 255, [3,3])


RGB_ew = np.random.normal(mu,sigma,[height, width, RGB])
RGB_eb = np.random.normal(mu,sigma,[height, width, 1])

w = np.random.normal(mu,sigma,[height, width, RGB])
b = np.random.normal(mu,sigma,[height//2, width//2, RGB])

ae_w1 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
ae_b1 = np.random.normal(mu,sigma,[height//4, width//4, RGB])

ae_w2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
ae_b2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])

ae_ww2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])
ae_bb2 = np.random.normal(mu,sigma,[height//2, width//2, RGB])

ww1 = np.random.normal(mu,sigma,[height, width, RGB])
bb1 = np.random.normal(mu,sigma,[height, width, RGB])

ww2 = np.random.normal(mu,sigma,[height, width, RGB])
bb2 = np.random.normal(mu,sigma,[height, width, RGB])

RGB_dw1 = np.random.normal(mu,sigma,[height, width, RGB])
RGB_db1 = np.random.normal(mu,sigma,[height, width, RGB])

RGB_dw2 = np.random.normal(mu,sigma,[height, width, RGB])
RGB_db2 = np.random.normal(mu,sigma,[height, width, RGB])

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
# feature2 = np.zeros([height//4, width//4, RGB])
# active2 = np.zeros([height//4, width//4, RGB])
#
# for h in range(RGB):
#     for x in range(width//4):
#         for y in range(height//4):
#             for m in range(2):
#                 for n in range(2):
#                     feature2[y][x][h] += active1[2*y+n][2*x+m][h] * ae_w1[2*y+n][2*x+m][h]
#
# for h in range(RGB):
#     for x in range(width//4):
#         for y in range(height//4):
#             active2[y][x][h] = relu(feature2[y][x][h] + ae_b1[y][x][h])
#
# feature3 = np.zeros([height // 2, width // 2, RGB])
# active3 = np.zeros([height // 2, width // 2, RGB])
#
# for h in range(RGB):
#     for x in range(width//4):
#         for y in range(height//4):
#             for m in range(2):
#                 for n in range(2):
#                     feature3[2*y+n][2*x+m][h] += active2[y][x][h] * ae_w2[2*y+n][2*x+m][h]
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             active3[y][x][h] = relu(feature3[y][x][h] + ae_b2[y][x][h])
#
#
# feature4 = np.zeros([height, width, RGB])
# active4 = np.zeros([height, width, RGB])
#
# for h in range(RGB):
#     for x in range(width//2):
#         for y in range(height//2):
#             for m in range(2):
#                 for n in range(2):
#                     feature4[2*y+n][2*x+m][h] += active3[y][x][h] * ww[2*y+n][2*x+m][h]
#
# for h in range(RGB):
#     for x in range(width):
#         for y in range(height):
#             active4[y][x][h] = relu(feature4[y][x][h] + bb[y][x][h])
#
# loss = np.sum((np.square(copy1 - active4)) / (width*height))
#
# print(np.sum(loss))
#
# plot.imshow(active4)
# plot.show()
#
# before = active4
#
# before_loss = loss

for epoch in range(500):
    print("epoch == {}".format(epoch))
    if epoch % 2 == 0:
        copy = copy1
        ae_dw2 = ae_w2
        ae_db2 = ae_b2
        ww = ww1
        bb = bb1
        RGB_dw = RGB_dw1
        RGB_db = RGB_db1
    else:
        copy = copy2
        ae_dw2 = ae_ww2
        ae_db2 = ae_bb2
        ww = ww2
        bb = bb2
        RGB_dw = RGB_dw2
        RGB_db = RGB_db2

    # plot.imshow(copy)
    # plot.show()

    rate = 0.0

    rgb_feature1 = np.zeros([height, width, 1])
    rgb_active1 = np.zeros([height, width, 1])

    for x in range(width):
        for y in range(height):
            for h in range(RGB):
                rgb_feature1[y][x][0] += copy[y][x][h] * RGB_ew[y][x][h]

    for x in range(width):
        for y in range(height):
            rgb_active1[y][x][0] = relu(rgb_feature1[y][x][0])

    feature1 = np.zeros([height // 2, width // 2, RGB])
    active1 = np.zeros([height // 2, width // 2, RGB])

    for h in range(1):
        for x in range(width // 2):
            for y in range(height // 2):
                for m in range(2):
                    for n in range(2):
                        feature1[y][x][h] += rgb_active1[2 * y + n][2 * x + m][h] * w[2 * y + n][2 * x + m][h]

    for h in range(1):
        for x in range(width // 2):
            for y in range(height // 2):
                active1[y][x][h] = relu_dropout(feature1[y][x][h] + b[y][x][h], rate)

    feature2 = np.zeros([height // 4, width // 4, RGB])
    active2 = np.zeros([height // 4, width // 4, RGB])

    for h in range(1):
        for x in range(width // 4):
            for y in range(height // 4):
                for m in range(2):
                    for n in range(2):
                        feature2[y][x][h] += active1[2 * y + n][2 * x + m][h] * ae_w1[2 * y + n][2 * x + m][h]

    for h in range(1):
        for x in range(width // 4):
            for y in range(height // 4):
                active2[y][x][h] = relu_dropout(feature2[y][x][h] + ae_b1[y][x][h], rate)

    feature3 = np.zeros([height // 2, width // 2, RGB])
    active3 = np.zeros([height // 2, width // 2, RGB])

    for h in range(1):
        for x in range(width // 4):
            for y in range(height // 4):
                for m in range(2):
                    for n in range(2):
                        feature3[2 * y + n][2 * x + m][h] += active2[y][x][h] * ae_dw2[2 * y + n][2 * x + m][h]

    for h in range(1):
        for x in range(width // 2):
            for y in range(height // 2):
                active3[y][x][h] = relu_dropout(feature3[y][x][h] + ae_db2[y][x][h], rate)

    feature4 = np.zeros([height, width, RGB])
    active4 = np.zeros([height, width, RGB])

    for h in range(1):
        for x in range(width // 2):
            for y in range(height // 2):
                for m in range(2):
                    for n in range(2):
                        feature4[2 * y + n][2 * x + m][h] += active3[y][x][h] * ww[2 * y + n][2 * x + m][h]

    for h in range(1):
        for x in range(width):
            for y in range(height):
                active4[y][x][h] = relu_dropout(feature4[y][x][h] + bb[y][x][h], rate)

    rgb_feature2 =np.zeros([height, width, RGB])
    rgb_active2 =np.zeros([height, width, RGB])

    for h in range(RGB):
        for x in range(width):
            for y in range(height):
                rgb_feature2[y][x][h] += active4[y][x][0] * RGB_dw[y][x][h]

    for h in range(RGB):
        for x in range(width):
            for y in range(height):
                rgb_active2[y][x][h] = relu_dropout(rgb_feature2[y][x][h] + RGB_db[y][x][h], rate)

    loss = np.sum((np.square(copy - rgb_active2)) / (width * height))

    print(np.sum(loss))

    dH = rgb_active2 - copy
    #
    # max_active = active2 / active2.max()
    #
    # show_img = active2 / max_active * 255
    #
    # show_img = show_img.astype('uint8')

    # plot.imshow(active1.astype('uint8'))
    # plot.show()
    # # plot.imsave("epoch{}_active1.jpg".format(epoch), active1.astype('uint8'))
    # plot.imshow(active2.astype('uint8'))
    # plot.show()
    # # plot.imsave("epoch{}_active2.jpg".format(epoch), active2.astype('uint8'))
    # plot.imshow(active3.astype('uint8'))
    # plot.show()
    # # plot.imsave("epoch{}_active3.jpg".format(epoch), active3.astype('uint8'))
    # plot.imshow(active4.astype('uint8'))
    # plot.show()
    if epoch % 2 == 0:
        plot.imshow(rgb_active2.astype('uint8'))
        plot.show()
        # plot.imshow(rgb_active2)
        # plot.show()

        plot.imsave("RGB/epoch_{}_LOSS_{}.jpg".format(epoch, loss), rgb_active2.astype('uint8'))

    dRGB_dDW = np.zeros([height, width, RGB])
    dRGB_dDB = np.zeros([height, width, RGB])

    for h in range(RGB):
        for x in range(width):
            for y in range(height):
                dRGB_dDW[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0]) * active4[y][x][0]

    for h in range(RGB):
        for x in range(width):
            for y in range(height):
                dRGB_dDB[y][x][h] = dH[y][x][h] * relu_prime(feature4[y][x][0])

    dH_decode_RGB = np.zeros([height, width, 1])

    for h in range(RGB):
        for x in range(width):
            for y in range(height):
                dH_decode_RGB[y][x][0] += dH[y][x][h] * relu_prime(rgb_feature2[y][x][h]) * RGB_dw[y][x][h]


    dWW = np.zeros([height, width, RGB])
    dBB = np.zeros([height, width, RGB])

    for h in range(1):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dWW[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * active3[y][x][h]

    for h in range(1):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dBB[2*y+n][2*x+m][h] = dH_decode_RGB[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h])

    dHH = np.zeros([height//2, width//2, RGB])

    for h in range(1):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dHH[y][x][h] += dH[2*y+n][2*x+m][h] * relu_prime(feature4[2*y+n][2*x+m][h]) * ww[2*y+n][2*x+m][h]

    dAE_W2 = np.zeros([height//2, width//2, RGB])
    dAE_B2 = np.zeros([height//2, width//2, RGB])

    for h in range(1):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_W2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * active2[y][x][h]

    for h in range(1):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_B2[2*y+n][2*x+m][h] = dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h])

    dHH2 = np.zeros([height//4, width//4, RGB])

    for h in range(1):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dHH[y][x][h] += dHH[2*y+n][2*x+m][h] * relu_prime(feature3[2*y+n][2*x+m][h]) * ae_dw2[2*y+n][2*x+m][h]

    dAE_W1 = np.zeros([height//2, width//2, RGB])
    dAE_B1 = np.zeros([height//4, width//4, RGB])

    for h in range(1):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_W1[2*y+n][2*x+m][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * active1[2*y+n][2*x+m][h]

    for h in range(1):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dAE_B1[y][x][h] = dHH2[y][x][h] * relu_prime(feature2[y][x][h])

    dHH3 = np.zeros([height // 2, width // 2, RGB])

    for h in range(1):
        for x in range(width//4):
            for y in range(height//4):
                for m in range(2):
                    for n in range(2):
                        dHH3[2*y+n][2*x+m][h] += dHH2[y][x][h] * relu_prime(feature2[y][x][h]) * ae_w1[2*y+n][2*x+m][h]

    dW = np.zeros([height, width, RGB])
    dB = np.zeros([height//2, width//2, RGB])

    for h in range(1):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dW[2*y+n][2*x+m][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h]) * rgb_active1[2*y+n][2*x+m][h]

    for h in range(1):
        for x in range(width//2):
            for y in range(height//2):
                for m in range(2):
                    for n in range(2):
                        dB[y][x][h] = dHH3[y][x][h] * relu_prime(feature1[y][x][h])

    dRGB_dEW = np.zeros([height, width, RGB])
    dRGB_dEB = np.zeros([height, width, 1])

    for h in range(RGB):
        for x in range(width):
            for y in range(height):
                dRGB_dEW[y][x][h] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0]) * copy[y][x][h]

    for h in range(1):
        for x in range(width):
            for y in range(height):
                dRGB_dEB[y][x][0] = dW[y][x][0] * relu_prime(rgb_feature1[y][x][0])


    w = w - LR * dW
    b = b - LR * dB

    ae_w1 = ae_w1 - LR * dAE_W1
    ae_b1 = ae_b1 - LR * dAE_B1

    ae_dw2 = ae_dw2 - LR * dAE_W2
    ae_db2 = ae_db2 - LR * dAE_B2

    ww = ww - LR * dWW
    bb = bb - LR * dBB

    RGB_ew = RGB_ew - LR * dRGB_dEW
    RGB_eb = RGB_eb - LR * dRGB_dEB

    RGB_dw = RGB_dw - LR * dRGB_dDW
    RGB_db = RGB_db - LR * dRGB_dDB

    # w = weight_clipping_11_3D(w)
    # ww = weight_clipping_11_3D(ww)
    w = weight_clipping_11_3D(w)
    ww = weight_clipping_11_3D(ww)

    ae_w1 = weight_clipping_11_3D(ae_w1)
    ae_dw2 = weight_clipping_11_3D(ae_dw2)

    RGB_ew = weight_clipping_11_3D(RGB_ew)
    RGB_dw = weight_clipping_11_3D(RGB_dw)

    w = weights_average_filter_3D_mask_101(w, 2, b)
    ww = weights_average_filter_3D_mask_101(ww,2, bb)

    ae_w1 = weights_average_filter_3D_mask_101(ae_w1,2, ae_b1)
    ae_dw2 = weights_average_filter_3D_mask_101(ae_dw2,2 , ae_db2)

    RGB_ew = weights_average_filter_3D_mask_101(RGB_ew,2, RGB_eb)
    RGB_dw = weights_average_filter_3D_mask_101(RGB_dw,2, RGB_db)

    # b = bias_clipping_1616_3D(b)
    # bb = bias_clipping_1616_3D(bb)
    #
    # ae_b1 = bias_clipping_1616_3D(ae_b1)
    # ae_b2 = bias_clipping_1616_3D(ae_b2)

    if epoch % 2 == 0:
        ae_w2 = ae_dw2
        ae_b2 = ae_db2
        ww1 = ww
        bb1 = bb

        RGB_dw1 = RGB_dw
        RGB_db1 = RGB_db
    else:
        ae_ww2 = ae_dw2
        ae_bb2 = ae_db2
        ww2 = ww
        bb2 = bb

        RGB_dw2 = RGB_dw
        RGB_db2 = RGB_db

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


