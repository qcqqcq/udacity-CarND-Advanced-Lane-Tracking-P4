**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[checker]: ./output_images/undistorted_checker.jpg "Calib"
[undistort]: ./output_images/undistorted_road.jpg "Undistort"
[warp_src]: ./output_images/warp_source.jpg "Warp Src"
[warp_src_zoom]: ./output_images/warp_source_zoomed.jpg "Warp Src Zoom"
[warp]: ./output_images/warped_road.jpg "Warp"

[sobel_mag]: ./output_images/sobel_mag.jpg "SobelMag"
[sobel_mag_bin]: ./output_images/sobel_mag_bin.jpg "SobelMagBin"
[sobel_dir]: ./output_images/sobel_dir.jpg "SobelDir"
[sobel_dir_bin]: ./output_images/sobel_dir_bin.jpg "SobelDirBin"
[sat]: ./output_images/saturation.jpg "Sat"
[sat_bin]: ./output_images/saturation_bin.jpg "SatBin"
[combo]: ./output_images/combined_bin.jpg "Combo"

[hist]: ./output_images/binary_histogram.jpg "hist"
[win]: ./output_images/windows_polynomial.jpg "hist"

[overlaid]: ./output_images/overlaid.jpg "ol"

## Overall Architecture

There were two main tasks for this project

* Code up a pipeline for taking in an image and returning an annotated image that has lane markings and some physical measurements
* Tune the parameters of the pipeline to produce good lane tracking on the project video (project_video.mp4) 

The parameter tuning and visualization is found in Jupyter notebook advanced_lane_lanes.ipynb while the main logic (backend) is in tracktools.py. Classes are as follows:


* ImageProcessor: Does all image processing
** Calibrates camera
** Undistorts images using calibration
** Perspective transforms

* Thresher:  Does all the tresholding
** Calculates Sobel and applies thresholds
** Converts to HSL and applies thresholds

* LaneTracker:  Tracks a lane given a binary image
** Searches for lanes in whole image using sliding window
** Searches for lanes in small area around previously found lanes
** Applies polynomial fit
** Calculates lane curvature

* LaneVerifier: Verifies lanes
** Uses history to smooth lane lines
** Ensures lane are roughly parallel


##Camera Calibration

This code is tracktools.ImageProcessor.calibrate\_camera() 

Checkerboard calibration images are given in ./camera\_cal/calibration*.jpg.  These are 9x6 flat boards so objects points are set accordingly and are the same for each image. For image points, openCV's findChessboardCorners is used. Distortion coefficients are then found for these pairs of object and images points using openCV's calibrateCamera function.

A wrapper is then built around openCV's undistort function to produce the following images:

![alt text][checker]
![alt text][undistort]


##Perspective Transform
A perspective transform takes the image from the vehcile view to a bird's eye view. To acheive this, "source" points must be identified which correspond to 4 points that form a rectangle in the bird's eye view. This only needs to be done once and can be applied to all other images as long as the camera is not moved.  Source point identification can be done using machine vision but here it is manually selected using trial and error.

The image from ./test_images/straight_lines1.jpg was used. Source points are shown below.  These points resulted in two parallel lines after applying the transform.

![alt text][warp_src] 
![alt text][warp_src_zoom] 

For the rest of the writeup, we use a more typical image with curved roads found in ./test_images/test1.jpg.  Here is the perpsective warp applied to it:


![alt text][warp] 

Also notice how this process also sets a region of interest.



##Thresholding to get a binary image

Now that image has been transformed to a bird's eye view (which also identifies a region of interest), lanes are identified.  In this approach, various transforms are applied and threshold values are chosen based on qualitative evaluation based on test images and the project video.  

The three transformations applied are
* Sobel magnitude
* Sobel direction
* Saturation level (after HSL conversion)

Sobel magnitude is calculated by applied Sobel in X and y directions (thus forming a 2D vector at every pixel) and taking the magnitude. The image below show the transformation:

![alt text][sobel_mag]

From the colorbar (and after viewing other test images), accepted values are determined to be in the range of [30-100].  Thresholding on this (within range = 1, outside range = 0) results in the following binary image:

![alt text][sobel_mag_bin]

The same procedure is done for the direction of the Sobel vectors:

![alt text][sobel_dir]

where accepted values are in [0-0.4], resulting in the following binary:

![alt text][sobel_dir_bin]

After converting to HSL space, thresholding was only performed for the saturation channel:

![alt text][sat]

and accepting only values [120,220] gives:

![alt text][sat_bin]

Combining the various binary images into a final image can be done a number of ways.  Here, the combined binary chosen to be the union of the Sobel magnitude binary and saturation binary.  The Sobel direction binary was deemed too noisy. The combined binary is shown below:

![alt text][combo]

A weighted average was also attempted but did not shown noticable improvement (not shown).



## Identifying which binary pixels are part of the lane lines

From the binary image, the lane lines become clear with occasional noise.  To fit a polynomial and guess the middle of the lane lines are, we have to further refine which pixels are lanes and which (left or right) lane the pixels belong to. To make an initial guess of the lane positions, a histogram of the binary values along each column is calculated.  The histgram only evaluates pixels in the lower half of the image.

![alt text][hist]

The x axis represents columns in the combined binary image and the y axis is the number of (binary) pixels are 1 (and not 0) within that row. From looking at the peaks of the histogram and dividing along the center (pixel 640 in the horizontal direction), the initial position (lowest in the image) of both lanes are determined.  From here, we do a window search, shown below and explained after the image:

![alt text][hist]

The windows heights are 40 pixels and widths are 150 pixels.  The first window is at the lowest part of the image and centered along the initial left and right values (determined from the histogram). All pixels within the lowest window are identified (and colored for visualization). The windows then shift up until they reach the highest pixels. With every shift up, the windows can be re-centered if more than 50 pixels are idntified in the window.  If so, the next window is centered around the mean of the column values of the current window. 

Once pixel values are identified, a second degree polynomial is fit along the values to determine the center of each lane (shown in yellow above).

Once this whole-image search is completed, in the next frame left and right lane pixels are identified if they simply lay within a margin (100 pixels to the left or 100 pixels to the right) of the polynomial fit. A new polynomial is then fit with these new pixel values.

### Radius of curvature

To calculate the radius of curvature, a conversion from pixels to meters is required. By assuming a standard lane width of 3.7 meters and counting horizontal pixels, it is determined that there are 3.7/672 pixels per meter in the horizontal direction.  By assuming a lane segment is 3 meters in the vertical direction, it is determined that there are 3/65 meters per pixel in the vertical direction.  By using these ratios to convert to meters and re-fitting the polynomial, the polynomial coefficients are used to calculat the radius of curvature for each lane in each frame, evaluated at the bottom of the frame. It was found that the measurement was quite sensitive and noisey and perhaps not reliable using this approach.


## Transforming back to the camera view and pipelining to overlay on a video

Before bringing the lane polynomial fits back to the image, some lane validation and smoothing steps are performed.  Every lane position is the average of the last 10 positions, which mediates jumping lane lines. There are also requirements to check that the lanes are roughly parallel and of an acceptable width. If these requirements are not met, the lane lines take on the those of the last good lane tracks.

Once lanes are verified, they warped back to the camera view and overlayed on top of the original image.  From here, assuming a lane width of 3.7 meters, we can also determine the position of the vehicle laterally within the lane.  Here, positive means to the right of the center.

![alt text][overlaid]

These steps are then placed in a pipeline and used to produce the following video:

[https://youtu.be/_2KKQbVfB2E]

