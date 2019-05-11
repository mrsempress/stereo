"""
It is for the camera calibration.
From GetCamearaParaments(), We can get the camera matrix, distortion coefficients, rotation and translation vectors etc.
And call undistortImg(), we can undistort the images.
ReprojectionError() gives a good estimation of just how exact is the found parameters.
This should be as close to zero as possible.
"""

import numpy as np
import cv2
import glob

# global variance
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objP = np.zeros((6 * 7, 3), np.float32)
objP[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # reshape(-1,2), row is according to col, and col is two

# Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPoints = []  # 2d points in image plane.


def DrawChessboard(img, gray, patternSize, storepath, objPoints, imgPoints):
    """
    :param img: The images that need draw chessboard
    :param gray: The gray image
    :param patternSize: Pattern size
    :param storepath: The path of output
    """

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, patternSize, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objPoints.append(objP)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        # Draw and display the corners
        img2 = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
        cv2.imshow('Draw chessboard', img2)
        cv2.imwrite(storepath, img2)
        cv2.waitKey(800)


def getCameraPara(gray, objPoints, imgPoints):
    """
    :param gray: The gray image
    :return: retval, camera matrix, distortion coefficients, rotation vectors, translation vectors
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    print("================================camera================================")
    print("retval:")
    print(ret)
    print("camera matrix:")
    print(mtx)
    print("distortion coefficients:")
    print(dist)
    print("rotation vectors :")
    print(rvecs)
    print("translation vectors :")
    print(tvecs)
    return ret, mtx, dist, rvecs, tvecs


def calibration(images, mtx, dist, rvecs, tvecs, path):
    """
    :param images: The images that need calibrate
    :param mtx: Camera matrix
    :param dist: Distortion coefficients
    :param rvecs: Rotation vectors
    :param tvecs: Translation vectors
    :param path: The path of output
    :return: 
    """
        
    for fname in images:
        x = fname[12:]
        img = cv2.imread(fname)
        # The origin image is gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        undistortImg(mtx, dist, img, path, x)


def undistort(img, newcameramtx, mtx, dist, roi, path, id):
    """
    :param img: The image object
    :param newcameramtx: New camera matrix
    :param mtx: Old camera matrix
    :param dist: Distortion coefficients
    :param roi: All-good-pixels region in the undistorted image.
    :param path: The path of output
    :param id: The name of image
    """
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(path + '1_' + id, dst)


def remapping(img, newcameramtx, mtx, dist, w, h, roi, path, id):
    """
    :param img: The image object
    :param newcameramtx: New camera matrix
    :param mtx: Old camera matrix
    :param dist:Distortion coefficients
    :param w: The width of image
    :param h: The height of image
    :param roi: All-good-pixels region in the undistorted image.
    :param path: The path of output
    :param id: The id of image
    """
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(path + '2_' + id, dst)


def undistortImg(mtx, dist, img, path, id):
    """
    :param mtx: Camera matrix
    :param dist: Distortion coefficients
    :param img: The image need undistort
    :param path: The path of output
    :param id: The id of image
    """

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Using cv2.undistort()
    # This is the shortest path. Just call the function and use ROI obtained above to crop the result.

    undistort(img, newcameramtx, mtx, dist, roi, path, id)

    # Using remapping
    # This is curved path. First find a mapping function from distorted image to undistorted image.
    # Then use the remap function.
    remapping(img, newcameramtx, mtx, dist, w, h, roi, path, id)


def ReprojectionError(objPoints, imgPoints, rvecs, tvecs, mtx, dist):
    """
    :param objPoints: 3d point in real world space
    :param imgPoints: 2d points in image plane.
    :param rvecs: Rotation vectors
    :param tvecs: Translation vectors
    :param mtx: Camera matrix
    :param dist: Distortion coefficients
    :return: Total error
    """
    mean_error = 0
    for i in range(len(objPoints)):
        imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgPoints[i], imgPoints2, cv2.NORM_L2) / len(imgPoints2)
        mean_error += error

    total_error = mean_error / len(objPoints)

    print("total error: ", total_error)


def main():
    global  objPoints
    global imgPoints
    Path = './data/left/*.jpg'
    images = glob.glob(Path)
    sorted(images)

    cv2.startWindowThread()
    for fname in images:
        x = fname[16:]
        img = cv2.imread(fname)

        # The origin image is gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        DrawChessboard(img, gray, (7, 6), 'output/calibration/drawchessleft' + x, objPoints, imgPoints)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = getCameraPara(gray, objPoints, imgPoints)

    path = 'output/calibration/calibresult'
    calibration(images, mtx, dist, rvecs, tvecs, path)
    ReprojectionError(objPoints, imgPoints, rvecs, tvecs, mtx, dist)


if __name__ == '__main__':
    main()
