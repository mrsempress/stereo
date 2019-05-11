"""
It is for the binocular calibrate
"""
import numpy as np
import cv2
from calibration import getCameraPara
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d point in real world space
imagePoints1 = []  # 2d points in image plane.
imagePoints2 = []  # 2d points in image plane.
w = 0
h = 0
corners1 = []
corners2 = []

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objP = np.zeros((6 * 7, 3), np.float32)
objP[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # reshape(-1,2), row is according to col, and col is two
#
# obj = np.zeros((9 * 6, 3), np.float32)
# obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# obj = obj * 25  # 25 mm  18.1

found1 = False
found2 = False

for i in range(1, 15):
    if i == 10:
        continue
    leftpath = './data/left/left' + ('0' if (i < 10) else '') + str(i) + '.jpg'
    rightpath = './data/right/right' + ('0' if (i < 10) else '') + str(i) + '.jpg'
    # print(leftpath)
    # print(rightpath)
    img1 = cv2.imread(leftpath)  # left
    img2 = cv2.imread(rightpath)  # right

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (w, h) = (7, 6)

    found1, corners1 = cv2.findChessboardCorners(gray1, (w, h), None)
    found2, corners2 = cv2.findChessboardCorners(gray2, (w, h), None)

    if found1:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img1, (w, h), corners1, found1)

    if found2:
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img2, (w, h), corners2, found2)

    cv2.imshow('image1 left', img1)
    cv2.waitKey(500)
    cv2.imshow('image2 right', img2)
    cv2.waitKey(500)
    cv2.imwrite('output/calibration_binocular/drawchess' + leftpath[12:], img1)
    cv2.imwrite('output/calibration_binocular/drawchess' + rightpath[13:], img2)

    if found1 != 0 and found2 != 0:
        imagePoints1.append(corners1)
        imagePoints2.append(corners2)
        object_points.append(objP)
cv2.destroyAllWindows()

(w, h) = gray1.shape[::-1]
# retl, mtx_left, dist_left, rvecsl, tvecsl = getCameraPara(gray1, object_points, imagePoints1)
# retr, mtx_right, dist_right, rvecsr, tvecsr = getCameraPara(gray2, object_points, imagePoints2)
retl, mtx_left, dist_left, rvecsl, tvecsl = cv2.calibrateCamera(object_points, imagePoints1, (w, h), None, None)
retr, mtx_right, dist_right, rvecsr, tvecsr = cv2.calibrateCamera(object_points, imagePoints2, (w, h), None, None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points,
                                                                                                 imagePoints1,
                                                                                                 imagePoints2,
                                                                                                 mtx_left, dist_left,
                                                                                                 mtx_right, dist_right,
                                                                                                 (w, h))
print("================================stereo================================")
print("Rotation matrix")
print(R)
print("Transformation matrix")
print(T)
print("Essential matrix")
print(E)
print("Fundamental matrix")
print(F)

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx_left, dist_left,
                                                                  mtx_right, dist_right, (w, h), R, T)

# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, (w, h), cv2.CV_16SC2)

right_map1, right_map2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, (w, h), cv2.CV_16SC2)

# only one image
leftpath = './data/left/left04.jpg'
rightpath = './data/right/right04.jpg'

img1 = cv2.imread(leftpath)  # left
img2 = cv2.imread(rightpath)  # right

h, w = img1.shape[:2]
newcameramtxl, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w, h), 1, (w, h))
img1 = cv2.undistort(img1, mtx_left, dist_left, None, newcameramtxl)

h, w = img2.shape[:2]
newcameramtxr, roi = cv2.getOptimalNewCameraMatrix(mtx_right, dist_right, (w, h), 1, (w, h))
img2 = cv2.undistort(img2, mtx_right, dist_right, None, newcameramtxr)

img1_rectified = cv2.remap(img1, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
img2_rectified = cv2.remap(img2, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)

cv2.imshow('image1 left', img1_rectified)
cv2.waitKey(500)
cv2.imwrite('output/calibration_binocular/rectifiedleft04.jpg', img1_rectified)

cv2.imshow('image2 right', img2_rectified)
cv2.waitKey(500)
cv2.imwrite('output/calibration_binocular/rectifiedright04.jpg', img2_rectified)

cv2.destroyAllWindows()

# for i in range(1, 15):
#     if i == 10:
#         continue
#     leftpath = './data/left/left' + ('0' if (i < 10) else '') + str(i) + '.jpg'
#     rightpath = './data/right/right' + ('0' if (i < 10) else '') + str(i) + '.jpg'
#     # leftpath = 'output/calibration_binocular/drawchessleft' + ('0' if (i < 10) else '') + str(i) + '.jpg'
#     # rightpath = 'output/calibration_binocular/drawchessright' + ('0' if (i < 10) else '') + str(i) + '.jpg'
#     # print(leftpath)
#     # print(rightpath)
#
#     img1 = cv2.imread(leftpath)  # left
#     img2 = cv2.imread(rightpath)  # right
#
#     # h, w = img1.shape[:2]
#     # newcameramtxl, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w, h), 1, (w, h))
#     # img1 = cv2.undistort(img1, mtx_left, dist_left, None, newcameramtxl)
#     #
#     # h, w = img2.shape[:2]
#     # newcameramtxr, roi = cv2.getOptimalNewCameraMatrix(mtx_right, dist_right, (w, h), 1, (w, h))
#     # img2 = cv2.undistort(img2, mtx_right, dist_right, None, newcameramtxr)
#
#     img1_rectified = cv2.remap(img1, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
#     img2_rectified = cv2.remap(img2, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)
#
#     cv2.imshow('image1 left', img1_rectified)
#     cv2.waitKey(500)
#     retval = cv2.imwrite('output/calibration_binocular/rectified' + leftpath[12:], img1_rectified)
#     # if retval:
#     #     print("Succeed")
#
#     cv2.imshow('image2 right', img2_rectified)
#     cv2.waitKey(500)
#     cv2.imwrite('output/calibration_binocular/rectified' + rightpath[13:], img2_rectified)
#
# cv2.destroyAllWindows()
