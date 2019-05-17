# stereo
About project Stereo

***Instructions:***

Simply download the entire `stereo` completely, and you need:

* Python 3.6
* Opencv 4.0

## Calibration.py

This document completes the 6,7 sub-question, implements two functions of camera calibration and image correction. 

***Instructions:***

Run `calibration.py`, you can see the calibrated picture in `output/calibration/`. 

***Results***

The result case (one randomly selected, the rest can be viewed by running the program):

**Image before correction:**

![left03](data/left/left03.jpg)

**Draw the chessboard:**

![](output/calibration/drawchessleft03.jpg)

**Image after correction:**

*Method 1:*

![calibresult1_03](output/calibration/calibresult1_left03.jpg)

*Method 2:*

![calibresult2_03](output/calibration/calibresult2_left03.jpg)

> Note: Some pictures cannot be corrected, and the size of the picture is 0.



## epilines.py

After learning the Epipolar geometry, this document implements the functions of finding the corresponding points, calculating the fundamental matrix, calculating the epipolar line and marking them according to the official OpenCV document. 

And tried two methods provided in OpenCV: brute force and FLANN.

> And we use this to test whether camera rectiﬁcationis is good or not

***Instructions:***

Run `calibration.py`, you can see the calibrated picture in `output/epilines/`. 

***Results***

*One of the left image:*

![epilines_left03](output/epilines/epilines_left04.jpg)

*Corresponding the right image:*

![epilines_right03](output/epilines/epilines_right04.jpg)

*Corresponding the matches image:*

![epilines_03](output/epilines/epilines_03.jpg)

However, there are some errors in the corresponding points, and some corresponding points are not on the board. The corresponding point relationship of the keyboard is correct, probably because the location is similar, but it is not the object we want to study.



## calibrate_binocular.py

This document is used to implement **Stereo calibration and rectiﬁcation**. 

***Instructions:***

Run `calibration.py`, you can see the calibrated and rectified picture in `output/calibration/`. 

***Results***

（The figure below is the same source file analysis as the corresponding point analysis chart before correction.）

*The left original picture:*

![](data/left/left04.jpg)

*The right original image:*

![](data/right/right04.jpg)

*calibration and rectiﬁcation:*

(Obviously, it succeed.)

![](output/calibration_binocular/rectifiedleft04.jpg)

![](output/calibration_binocular/rectifiedright04.jpg)

Some values about the parameters are as follows：

![](output/calibration_binocular/stereo_calibration.png)



After stereo correction, perform corresponding point analysis:

![](output/calibration_binocular/epilines_left04.jpg)

![](output/calibration_binocular/epilines_right04.jpg)

We can find that the epipolar lines are parallel and the corresponding pixels are the same position in the two figures. It may prove that the rectiﬁcation is successful.

## Disparity

In this module, I use OpenCV to compute the disparity maps for the images I used for stereo calibration. And the result as follows.

### Disparity_BM.py

Block matching method:

*The left image:*

![](output/Disparity/BMleft104.jpg)

*The right image:*

![BMright04](output/Disparity/BMright104.jpg)

### Disparity_DP.py

Dynamic programming method:

*The left image:*

![](output/Disparity/DPleft04.jpg)

*The right image:*

![DPright04](output/Disparity/DPright04.jpg)

### Disparity_SGBM.py

Semi-global block matching method:

*The left image:*

![SGBMleft104](output/Disparity/SGBMleft104.jpg)

*The right image:*

![SGBMright104](output/Disparity/SGBMright104.jpg)

当增加SGBM中的blocksize值后，可以减少一些噪声，但是也会去除一些有用的信息：

![](output/Disparity/SGBMleft204.jpg)

![SGBMright204](output/Disparity/SGBMright204.jpg)

# Modify 

1. 2019/05/03	First upload, completed camera calibration and image correction 
2. 2019/05/04	Fixed the difference between the generated image and the original image (because the sequence returned by `glob.glob()` is not based on the named alphabetical order)
3. 2019/05/09	Add `epilines.py`
4. 2019/05/10	Add `calibrate_binocular.py`
5. 2019/05/11	Modify `calibrate_binocular.py`, add  *rectiﬁcation* function
6. 2019/05/12	Add `calibrate_binocular.m`, which is from [Tutorial on Rectification of Stereo Images]([http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO/node18.html](http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO/node18.html))
7. 2019/05/14	Add `Disparity_BM.py`, `Disparity_DP.py`, `Disparity_SGBM.py`
8. 2019/05/15	Add timing function
9. 2019/05/17	Temporarily stop project update



