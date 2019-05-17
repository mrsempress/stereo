import cv2
import numpy as np
from matplotlib import pyplot as plt
# disparity settings
min_disp = 32
num_disp = 112 - min_disp

# do it
imgL = cv2.imread('output/calibration_binocular/rectifiedleft04.jpg', 0)
imgR = cv2.imread('output/calibration_binocular/rectifiedright04.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
disparity = (disparity-min_disp)/num_disp
print("========================Left Disparity=========================")
print(disparity)
plt.imshow(disparity, 'gray')
plt.savefig('output/Disparity/BMleft104.jpg')
plt.show()

# cv2.imwrite('output/Disparity/BMleft04.jpg', disparity)


disparity = stereo.compute(imgR, imgL).astype(np.float32) / 16.0
disparity = (disparity-min_disp)/num_disp
print("========================Left Disparity=========================")
print(disparity)
plt.imshow(disparity, 'gray')
plt.savefig('output/Disparity/BMright104.jpg')
plt.show()
# cv2.imwrite('output/Disparity/BMright04.jpg', disparity)
