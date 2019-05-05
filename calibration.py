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


def GetCameraParaments(objP, Path, patternSize, iteration=30, epsilon=0.001):
    """
    :param objP: 3d point in real world space
    :param Path: The path of images
    :param patternSize: Pattern size
    :param iteration: The max Iteration of the corner position moves
    :param epsilon: The max Epsilon of the corner position moves
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iteration, epsilon)

    # Arrays to store object points and image points from all the images.
    objPoints = []  # 3d point in real world space
    imgPoints = []  # 2d points in image plane.

    images = glob.glob(Path)
    sorted(images)
    cv2.startWindowThread()
    for fname in images:
        x = fname[12:]
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, patternSize, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objPoints.append(objP)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgPoints.append(corners2)

            # Draw and display the corners
            img2 = cv2.drawChessboardCorners(img, patternSize, corners2, ret)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
            print("===========================================================================")
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

            cv2.imshow('img', img2)
            undistortImg(mtx, dist, img, x)
            ReprojectionError(objPoints, imgPoints, rvecs, tvecs, mtx, dist)

            cv2.waitKey(0)
    cv2.destroyAllWindows()


def undistort(img, newcameramtx, mtx, dist, roi, id):
    """
    :param img: The image object
    :param newcameramtx: New camera matrix
    :param mtx: Old camera matrix
    :param dist:Distortion coefficients
    :param roi: All-good-pixels region in the undistorted image.
    :param id: The name of image
    """
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('output/calibresult1_' + id, dst)

def remapping(img, newcameramtx, mtx, dist, w, h, roi, id):
    """
    :param img: The image object
    :param newcameramtx: New camera matrix
    :param mtx: Old camera matrix
    :param dist:Distortion coefficients
    :param w: The width of image
    :param h: The height of image
    :param roi: All-good-pixels region in the undistorted image.
    :param id: The id of image
    """
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('output/calibresult2_' + id, dst)


def undistortImg(mtx, dist, img, x):
    """
    :param mtx: Camera matrix
    :param dist: Distortion coefficients
    :param img: The image need undistort
    """

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Using cv2.undistort()
    # This is the shortest path. Just call the function and use ROI obtained above to crop the result.

    undistort(img, newcameramtx, mtx, dist, roi, x)

    # Using remapping
    # This is curved path. First find a mapping function from distorted image to undistorted image.
    # Then use the remap function.
    remapping(img, newcameramtx, mtx, dist, w, h, roi, x)


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
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objP = np.zeros((6 * 7, 3), np.float32)
    objP[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # reshape(-1,2), row is according to col, and col is two

    GetCameraParaments(objP, './data/left/*.jpg', (7, 6))


if __name__ == '__main__':
    main()
