import numpy as np
import cv2

img = cv2.imread('C://cv_learn/desk.jpg',1)

img = cv2.resize(img, (600,400),cv2.INTER_CUBIC)

#convert image to grey scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sobel operator
#define kernels
kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#convolution func
def conv(img, kernel_x,kernel_y):
    row,col = img.shape[:2]
    result_x = np.empty(((1,0)))
    result_y = np.empty(((1,0)))
    for r in range(0,row-2):
        for c in range(0,col-2):
            window = img[r:r+3,c:c+3]
            temp_x = np.multiply(kernel_x,window)
            temp_y = np.multiply(kernel_y,window)
            temp_x = np.array([[temp_x.sum()]])
            temp_y = np.array([[temp_y.sum()]])
            result_x = np.append(result_x,temp_x,1)
            result_y = np.append(result_y,temp_y,1)
        print(r)
    return (result_x,result_y)

#padding added
img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value=255)
#convolution
# conv_x = conv(img, kernel_x)
# img_conv_x = conv_x.reshape(400,600)
# conv_y = conv(img, kernel_y)
# img_conv_y = conv_y.reshape(400,600)

conv_x,conv_y = conv(img,kernel_x,kernel_y)
img_conv_x = np.abs(conv_x.reshape(400,600))
img_conv_y = np.abs(conv_y.reshape(400,600))

IMG = (np.abs(conv_x) + np.abs(conv_y)).reshape((400,600))


cv2.imshow('image1',img_conv_x)
#show 
cv2.imshow('image2',img_conv_y)
cv2.imshow('image3',IMG)
cv2.imshow('image', img)

k = cv2.waitKey(0)
if k == 27:
     cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('C://cv_learn/sobel_x.png',img_conv_x)
    cv2.imwrite('C://cv_learn/sobel_y.png',img_conv_y)
    cv2.imwrite('C://cv_learn/sobel.png',IMG)
    cv2.destroyAllWindows()

