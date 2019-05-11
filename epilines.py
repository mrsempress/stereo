"""
It is for Epipolar geometry
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def Epipolar_geometry(leftpath, rightpath):
    """
    :param leftpath: The path of left images
    :param rightpath: The path of right images
    :return:
    """
    # objP = np.zeros((6 * 7, 3), np.float32)
    # objP[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) 
    # patternSize = (7, 6)
    imgl = cv2.imread(leftpath, 0)  # queryimage # left image
    imgr = cv2.imread(rightpath, 0)  # trainimage # right image
    # id = leftpath[16:]
    id = leftpath[42:]
    # # The origin image is gray
    # grayl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
    # grayr = cv2.cvtColor(imgr,cv2.COLOR_BGR2GRAY)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # retl, cornersl = cv2.findChessboardCorners(grayl, patternSize, None)
    # retr, cornersr = cv2.findChessboardCorners(grayr, patternSize, None)
    # if not retl or not retr:
    #     return

    # cornersl2 = cv2.cornerSubPix(grayl, cornersl, (11, 11), (-1, -1), criteria)
    # cornersr2 = cv2.cornerSubPix(grayr, cornersr, (11, 11), (-1, -1), criteria)

    # imgl = cv2.drawChessboardCorners(grayl, patternSize, cornersl2, retl)
    # imgr = cv2.drawChessboardCorners(grayr, patternSize, cornersr2, retr)

    # FLANN: Fast Libary for Approximate Nearest Neighbors
    (pts1, pts2) = findMatches(imgl, imgr, id)
    F, pts1, pts2 = findFundamentalMatrix(pts1, pts2)
    findEpilines(imgl, imgr, pts1, pts2, F, id)

    # # Brute Force
    # sift = cv2.xfeatures2d.SIFT_create(100)
    # kp1, des1 = sift.detectAndCompute(imgl, None)
    # kp2, des2 = sift.detectAndCompute(imgr, None)
    # bf = cv2.BFMatcher()
    # # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
    # matches = bf.knnMatch(des1, des2, k=2)
    # goodMatches = []
    # minRatio = 1/3
    # for m,n in matches:
    #     if m.distance / n.distance < minRatio:   
    #         goodMatches.append([m])
    # sorted(goodMatches,key=lambda x:x[0].distance)
    # #绘制最优匹配点
    # img3 = None
    # img3 = cv2.drawMatchesKnn(imgl, kp1, imgr, kp2, matches, img3, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # img3 = cv2.resize(img3,(1000, 400))
    # cv2.imwrite('output/epilines/epilines_' + id, img3)


def findMatches(img1, img2, id):
    """
    :param img1: The left image
    :param img2: The right image
    :param id: The name of image
    :return: The list of symmetric point
    """
    # vgg = cv2.xfeatures2d.VGG_create()
    # brisk = cv2.BRISK_create()
    # gms = cv2.xfeatures2d.matchGMS()
    # sift = cv2.xfeatures2d.SIFT_create(100)
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

            # draw matches
    drawParams = dict(  # singlePointColor=(255,0,0), matchColor=(0,255,0),
        matchesMask=matchesMask,
        flags=0)
    resultImage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawParams)

    # # Univariate transformation
    # matchesMask = Univariatetrans(good, kp1, kp2, img1, img2)
    # # draw matches
    # drawParams = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None, matchesMask = matchesMask, flags = 2)
    # resultImage = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **drawParams)

    # cv2.imwrite('output/epilines/epilines_' + id, resultImage)
    cv2.imwrite('output/calibration_binocular/epilines_' + id, resultImage)

    return pts1, pts2


def Univariatetrans(goodMatches, kp1, kp2, img1, img2):
    """
    :param goodMatches: The matches points
    :param kp1: keypoints 1
    :param kp2: keypoints 2
    :param img1: image 1
    :param img2: image 2
    :return: matchesMask
    """
    MIN_MATCH_COUNT = 10

    if len(goodMatches) > MIN_MATCH_COUNT:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 2)

        # Get the projection matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()  # 用来配置匹配图，只绘制单应性图片中关键点的匹配线

        h, w = img1.shape[:2]

        # four corner
        pts = np.float32([[55, 74], [695, 45], [727, 464], [102, 548]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Draw the framework
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
        matchesMask = None

    return matchesMask


def findFundamentalMatrix(pts1, pts2):
    """
    :param pts1: Symmetric point list 1
    :param pts2: Symmetric point list 2
    :return: Fundamental matrix and inlier points
    """
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 5.0)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return F, pts1, pts2


def findEpilines(img1, img2, pts1, pts2, F, id):
    """
    :param img1: The left image
    :param img2: The right image
    :param pts1: Symmetric point 1
    :param pts2: Symmetric point 2
    :param F: Fundamental matrix
    :param id: The id of raw picture
    :return:
    """
    # Find epilines corresponding to points in right image (second image) [img6] and
    # drawing its lines on left image [img5]
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) [img4] and
    # drawing its lines on right image [img3]
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    # cv2.imwrite('output/epilines/epilines_left' + id, img5)
    # cv2.imwrite('output/epilines/epilines_right' + id, img3)
    cv2.imwrite('output/calibration_binocular/epilines_left' + id, img5)
    cv2.imwrite('output/calibration_binocular/epilines_right' + id, img3)
    # plt.subplot(121), plt.imshow(img5)
    # plt.subplot(122), plt.imshow(img3)

    plt.subplot(221), plt.imshow(img5)
    plt.subplot(222), plt.imshow(img6)
    plt.subplot(223), plt.imshow(img3)
    plt.subplot(224), plt.imshow(img4)
    plt.show()


def drawlines(img1, img2, lines, pts1, pts2):
    """
    :param img1: The image on which we draw the epilines for the points in img2
    :param img2: The other image
    :param lines: corresponding epilines
    :param pts1: Inlier point 1
    :param pts2: Inlier point 2
    :return: The new left and right image
    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def main():
    for id in range(1, 15):
        if id == 10:
            continue
        # leftpath = './data/left/left' + ('0' if (id < 10) else '') + str(id) + '.jpg'
        # rightpath = './data/right/right' + ('0' if (id < 10) else '') + str(id) + '.jpg'
        leftpath = 'output/calibration_binocular/rectifiedleft' + ('0' if (id < 10) else '') + str(id) + '.jpg'
        rightpath = 'output/calibration_binocular/rectifiedright' + ('0' if (id < 10) else '') + str(id) + '.jpg'
        print(leftpath)
        print(rightpath)
        Epipolar_geometry(leftpath, rightpath)


if __name__ == '__main__':
    main()
