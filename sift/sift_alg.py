import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from math import pi, exp



# generate gaussian filter with given size and s.d.

class gaussian_filter():
    def __init__(self, size, sd):
        self.size = size 
        self.sd = sd
        if(size % 2 == 0):
            raise ValueError("invalid kernal size.")

        self.kernal = self.gen_kernal()

    def gaussian_2d(self, coord):
        x = coord[0]
        y = coord[1]
        return (1/(2*pi*self.sd**2))*(exp(-(x**2 + y**2)/(2*self.sd**2)))

    def get_recentered_coord(self, coord):
        x = coord[0] - int(self.size/2)
        y = coord[1] - int(self.size/2)
        return (x,y)

    def gen_kernal(self):
        kernal = np.zeros((self.size, self.size))
        for x in range(kernal.shape[0]):
            for y in range(kernal.shape[1]):
                coord = self.get_recentered_coord((x,y))
                kernal[x][y] = self.gaussian_2d(coord)
        
        kernal = kernal/np.abs(np.sum(kernal))
        print (np.sum(kernal))
        return kernal
    
    def plot_kernal(self):
        x_s = np.array(range(self.kernal.shape[0]))-self.size
        y_s = np.array(range(self.kernal.shape[1]))-self.size
        x_grid, y_grid = np.meshgrid(x_s, y_s)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        ax.plot_surface(x_grid, y_grid, self.kernal, rstride = 1, 
                        cstride = 1, cmap = plt.cm.coolwarm)
        plt.show()


class laplacian_of_gaussian_filter():
    def __init__(self,size, sd):
        self.size = size 
        self.sd = sd
        if(size % 2 == 0):
            raise ValueError("invalid kernal size.")

        self.kernal = self.gen_kernal()
        return

    def LoG_2d(self, coord):
        x = coord[0]
        y = coord[1]
        return (-1/(pi*self.sd**4))*(exp(-1*(x**2+y**2)/2/self.sd**2))*(1-(x**2+y**2)/2/self.sd**2)
    
    def get_recentered_coord(self, coord):
        x = coord[0] - int(self.size/2)
        y = coord[1] - int(self.size/2)
        return (x,y)
    
    def gen_kernal(self):
        kernal = np.zeros((self.size, self.size))
        for x in range(kernal.shape[0]):
            for y in range(kernal.shape[1]):
                coord = self.get_recentered_coord((x,y))
                kernal[x][y] = self.LoG_2d(coord)
        
        # kernal = kernal/np.abs(np.sum(kernal))
        print (np.sum(kernal))
        return kernal
    
    def plot_kernal(self):
        x_s = np.array(range(self.kernal.shape[0]))-self.size
        y_s = np.array(range(self.kernal.shape[1]))-self.size
        x_grid, y_grid = np.meshgrid(x_s, y_s)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        ax.plot_surface(x_grid, y_grid, self.kernal, rstride = 1, 
                        cstride = 1, cmap = plt.cm.coolwarm)
        plt.show()






def img_convolve(img, img_filter):  #mode == constant
    color_chanels = len(img.shape)
    border_size = int(img_filter.size/2)
    bordered_img = cv2.copyMakeBorder(img, border_size, 
                    border_size, border_size, border_size, 
                    cv2.BORDER_CONSTANT, value = 0)
    output = np.zeros(img.shape)
    if color_chanels == 2:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                area_of_interest = bordered_img[x:x + img_filter.size,
                                                y:y + img_filter.size] 
                cov_result = np.sum(area_of_interest * img_filter.kernal)
                output[x][y] = cov_result
                loading_bar(x*img.shape[1]+y, (img.shape[0]-1) * (img.shape[1]-1))
    elif color_chanels == 3:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                area_of_interest = bordered_img[x:x + img_filter.size,
                                                y:y + img_filter.size] 
                cov_result_0 = np.sum(area_of_interest[:,:,0] 
                                        * img_filter.kernal)
                cov_result_1 = np.sum(area_of_interest[:,:,1]  
                                        * img_filter.kernal)
                cov_result_2 = np.sum(area_of_interest[:,:,2] 
                                        * img_filter.kernal)
                output[x][y][0] = cov_result_0
                output[x][y][1] = cov_result_1
                output[x][y][2] = cov_result_2
                loading_bar(x*img.shape[1]+y, (img.shape[0]-1) * (img.shape[1]-1))
    return output


def loading_bar(value, upper_limit):
    percentage = round(value/upper_limit * 100,2)
    max_len = 50
    cur_len = int(max_len * percentage/100)
    filler = " "*(5-len(str(percentage)))
    bar = "="*cur_len
    space = " "*(max_len-cur_len)
    if value % 20000 == 0 or value == upper_limit:
        display = "[{0}{1}%] [{2}{3}]".format(filler,percentage, bar, space)
        print(display)

def img2uint8(image):
    return np.uint8(image)



if __name__ == "__main__":

    pathname = os.path.dirname(os.path.realpath(__file__))

    img = mpimg.imread(pathname + '/../queen.jpg')
    resized_img = cv2.resize(img, (1200,800), cv2.INTER_CUBIC)
    print(resized_img.shape)

    # img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Gaussian
    gauss_f = gaussian_filter(5, 1)
    # print (gauss_f.kernal.shape)
    gauss_f.plot_kernal()
    print(resized_img[:,:,1].shape) 
    gaussian_img = img_convolve(resized_img[:,:,:], gauss_f)

    # gaussian_img = fast_img_convolve(resized_img[:,:,1], gauss_f)

    # gaussian_img_cv = cv2.GaussianBlur(resized_img[:,:,1], (21, 21), 5, 5)
    # gaussian_diff = gaussian_img - gaussian_img_cv
    # LoG
    LoG_f = laplacian_of_gaussian_filter(41,2)
    LoG_f.plot_kernal()

    LoG_image = img_convolve(resized_img[:,:,1], LoG_f)

    """plotting"""
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ### 1.  cv2 default = BGR,  matplotlib.imshow default = RGB
    ### 2.  img need to be unit8 encoded!

    # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # convolved_img = np.uint8(convolved_img) 
    # convolved_img = cv2.cvtColor(convolved_img, cv2.COLOR_BGR2RGB)

    ax1.imshow(resized_img)
    # ax2.imshow(convolved_img, cmap = plt.cm.gray)
    ax2.imshow(img2uint8(gaussian_img))
    # ax3.imshow(img2uint8(LoG_image))

    # ax2.imshow(img2uint8(gaussian_img), cmap = plt.cm.Greys)
    ax3.imshow(img2uint8(LoG_image), cmap = plt.cm.Greys)

    plt.show()
