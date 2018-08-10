import numpy as np
import cv2
import os
#import image


# read in image, 0 greyscale, 1 color, -1 unchanged
img = cv2.imread('C://cv_learn/desk.jpg',1)



#putting text on image
# img_1 = cv2.imread('c://cv_learn/desk.jpg',cv2.IMREAD_COLOR)
# img_1t = cv2.putText(img_1,'text on image',(1000,1000),3,10,(255,255,255),10)
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.imshow('image',img_1t)
# k = cv2.waitKey(0)
# if k == 27:
#     cv2.destroyAllWindows()
# elif k == ord('s'):
#     cv2.imwrite('C://cv_learn/desk_text.png',img_1t)
#     cv2.destroyAllWindows()


#resize image
res = cv2.resize(img,(600,400),cv2.INTER_LINEAR)
img = res
# show = res


#Image translation
# img = res
# col,row = img.shape[:2]
# M = np.float32([[1,0,1],[0,1,0]])
# dst = cv2.warpAffine(img,M,(col,row))
# show = dst


#color change
# col = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# show = col
# lower_blu = np.array([110,50,50])
# upper_blu = np.array([130,255,255])
# mask = cv2.inRange(col, lower_blu, upper_blu)
# show = mask
# res = cv2.bitwise_and(img,img,None,mask)
# show = res

#plit/merge of col channle
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))
show = img

#show the image
#save the image
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',show)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('C://cv_learn/desk_text.png',show)
    cv2.destroyAllWindows()