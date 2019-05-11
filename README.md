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

*Rectify:*

(Obviously, it succeed.)

![](output/calibration_binocular/rectifiedleft04.jpg)

![](output/calibration_binocular/rectifiedright04.jpg)

Some values about the parameters are as follows：

![](output/calibration_binocular/stereo_calibration.png)



After stereo correction, perform corresponding point analysis:

![](output/calibration_binocular/epilines_left04.jpg)

![](output/calibration_binocular/epilines_right04.jpg)

We can find that the epipolar lines are parallel and prove that the rectiﬁcation is successful.

# Modify 

1. 2019/05/03	First upload, completed camera calibration and image correction 
2. 2019/05/04    Fixed the difference between the generated image and the original image (because the sequence returned by `glob.glob()` is not based on the named alphabetical order)
3. 2019/05/09    Add `epilines.py`
4. 2019/05/10    Add `calibrate_binocular.py`
5. 2019/05/11    Modify `calibrate_binocular.py`, add  *rectiﬁcation* function

