
import cv2

# 读取图像
# image = cv2.imread('test.jpeg')

# 显示图像
# cv2.imshow('Original Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# 3.特征提取SIFT
# cv2.SIFT_create()    #创建一个sift特征的提取对象
# sift.detect(img)        #在图像中查找关键点
zhang = cv2.imread('test.jpeg')
sift = cv2.SIFT_create()  #创建角点检测
kp = sift.detect(zhang)   #对图像进行角点检测，得到关键点
zhang_sift=cv2.drawKeypoints(zhang,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('zhang_sift', zhang_sift)
cv2.waitKey(0)

