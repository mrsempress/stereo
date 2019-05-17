import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

start = time.clock()

# disparity settings
min_disp = 32
num_disp = 112 - min_disp
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)


# do it
imgL = cv2.imread('output/calibration_binocular/rectifiedleft04.jpg', 0)
imgR = cv2.imread('output/calibration_binocular/rectifiedright04.jpg', 0)

# disparity = stereo.compute(imgL, imgR)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
print("========================Left Disparity=========================")
print(disparity)
plt.imshow(disparity, 'gray')
plt.savefig('output/Disparity/SGBMleft104.jpg')
plt.show()
print("========================Left Depth=========================")
disparity = (disparity-min_disp)/num_disp
print(disparity)
plt.imshow(disparity, 'gray')
plt.savefig('output/Depth/SGBMleft104.jpg')
plt.show()

# cv2.imwrite('output/Disparity/SGBMleft04.jpg', disparity)


# disparity = stereo.compute(imgL, imgR)
disparity = stereo.compute(imgR, imgL).astype(np.float32) / 16.0
print("========================Right Disparity=========================")
print(disparity)
plt.imshow(disparity, 'gray')
plt.savefig('output/Disparity/SGBMright104.jpg')
plt.show()
print("========================Right Depth=========================")
disparity = (disparity-min_disp)/num_disp
print(disparity)
plt.imshow(disparity, 'gray')
plt.savefig('output/Depth/SGBMright104.jpg')
plt.show()

# cv2.imwrite('output/Disparity/SGBMright04.jpg', disparity)

end = time.clock()
print("time:" + str(end - start))
