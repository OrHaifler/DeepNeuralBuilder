import numpy as np
import cv2


def procces(name):
    image = cv2.imread(name)
    #image = cv2.cvtColor(src = image, code=cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('test.jpeg', image)
    #test = cv2.imread('test.jpeg')
    return image.transpose(2, 0, 1)


def convolve(image, kernel, padding = 0):

    padded = np.pad(image, ((0,0), (padding,padding), (padding,padding)), 'constant')
    print(padded[0])
    print(padded[1])
    print(padded[2])
    if len(kernel.shape) > 2:
        C = kernel.shape[0]
        if(image.shape[0] != C):
            return "Input channels error"
    Kx, Ky = kernel.shape[1:]
    Ix, Iy = image.shape[1:]

    outX = 2 * padding + Ix - Kx + 1
    outY = 2 * padding + Iy - Ky + 1
    output = np.zeros((C, outX, outY))

    for i in range(outX):
        for j in range(outY):
            for c in range(C):
                output[c, i, j] = (padded[c, i:i+Kx, j:j+Ky] * kernel[c]).sum()
    print(output[0])
    print(output[1])
    print(output[2])
    return output.transpose(1, 2, 0)



image = procces('buildings.jpeg')
print(image.shape)
zeros = np.zeros((3,3))
kernel1d = np.random.normal(1000, 1, size = (5,5))
kernel = np.array([kernel1d, kernel1d, kernel1d]) * 1 / kernel1d.sum()
output = convolve(image, kernel, padding=0)
cv2.imwrite('convolved.jpeg', output)
out = cv2.imread('convolved.jpeg')
cv2.imshow('convolved', out)
cv2.waitKey(0)
