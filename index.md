---
title: Lane Tracking for Autonomous Vehicles
tagline: Tracking lanes using classical machine vision for the Udacity AV Nanodegree Program
description: Tracking lanes using classical machine vision for the Udacity AV Nanodegree Program
---

*This is part of my series on the Udacity Self Driving Car Nanodegree Program*

One of the cool things about autonomous vehicle technlogies is that it takes ideas from all fields. While deep neural networks are all the rage nowadays, classical machine vision still has it's place in the field. Will this still be the case in the next 30 years?  Only time will tell.  In the meanwhile, check out what I did using a simple workflow with Sobel operators and a few transforms identify and track lane lines. I'll start out showing the end results first:


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/_2KKQbVfB2E/0.jpg)](https://www.youtube.com/watch?v=_2KKQbVfB2E)


If clicking the above image doesn't take you to the video, trying using the [Video Link Here](https://youtu.be/_2KKQbVfB2E)

## Overall Steps

To create the video, the following steps were taken:

* Calibrate the camera using given a set of chessboard images to correct for radial and tangental distortions.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify the image ("birds-eye view").
* Use color transforms, gradients, etc. and apply thresholds to obtain a binary image
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

## Camera Calibration

Checkerboard calibration images are given in the repository  They tell the camera what a flat, straight, perpendicular lines look like so it can apply the appropriate corrections.  Check out how it corrects the below image:

![alt text][checker]

Notice how in the left image, you can intuitively tell that something's not quite right. If you look closely, you'll notice the checkerboard lines are quite...straight.  They seem "bent" somehow.  This is kind of [distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) is actually present in almost any lens out there. You probably didn't notice since today's high tech cameras where the lens position never changes already correct for this. Also, you probably aren't taking pictures of checkerboards so it's not actually that easy to tell.

Below is the correction applied to a real world image.  See any differences?

![alt text][undistort]

I can't tell either.  I guess maybe if I take out the photo editor and overlay some lines maybe I can tell. But for most artistic pictures of the kids, this kind of thing probably isn't noticable.  BUT, since we're doing careful lane tracking here, we need to be a bit more careful.

## Perspective Transform

A perspective transform takes the image from the vehcile view to a bird's eye view. To acheive this, "source" points must be identified which correspond to 4 points that form a rectangle in the bird's eye view. This only needs to be done once and can be applied to all other images as long as the camera is not moved.  Source point identification can be done using machine vision but here it is manually selected using trial and error.

A test image with straight lanes is used. Source points are shown below.  These points resulted in two parallel lines after applying the transform.

![alt text][warp_src] 
![alt text][warp_src_zoom] 

Here's the perpsective transform applied to a more curvey road:

![alt text][warp] 

Also notice how this process also sets a region of interest.  It allows us to ignore anything outside of this bird's eye view. This comes in handy when we're applying transforms and deciding on which threshold values to use.

## Thresholding to get a binary image

Now that image has been transformed to a bird's eye view (which also identifies a region of interest), lanes are identified.  In this approach, various transforms are applied and threshold values are chosen based on qualitative evaluation based on test images and the project video. We then form binary images where if the transformed pixel has values within the threshold range, we set that pixel to 1, otherwise set it to 0.  This gives a binary image.  Since might seem unclear at first if you're not from the machine vision world, so I'll outine the steps below.

The three transformations applied are
* Sobel magnitude
* Sobel direction
* Saturation level (after HSL conversion)


### Sobel Magnitude

Sobel magnitude is calculated by converting the image to gray scale (i.e. a black and white image) then applying the [Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator) in X and y directions (thus forming a 2D vector at every pixel) and taking the magnitude. This vector is the gradient of the pixel intensities. The image below show the transformation:

![alt text][sobel_mag]

Here the colorbar values are the magnitutes of the Sobel vectors. After viewing a bunch of images and experimenting with different threshold values, I chose to accept only pixels in the range of [30-100].  Thresholding on this (within range = 1, outside range = 0) results in the following binary image:

![alt text][sobel_mag_bin]


### Sobel Direction

The same procedure is done for the direction of the Sobel vectors:

![alt text][sobel_dir]

where accepted values are in [0-0.4], resulting in the following binary:

![alt text][sobel_dir_bin]


### Color Saturation
Normally, images are described in terms of the pixel's RGB (red-green-blue) values. But for image processing, this is isn't always the best respresentation.  Here, we first convert the image to HSL (hue-saturation-luminance) space, then threshold only based on the saturation channel:

![alt text][sat]

and accepting only values [120,220] gives:

![alt text][sat_bin]


### Combining Binary Images
Combining the various binary images into a final image can be done a number of ways.  Here, the combined binary chosen to be the union of the Sobel magnitude binary and saturation binary.  The Sobel direction binary was deemed too noisy. The combined binary is shown below:

![alt text][combo]


## Identifying which binary pixels are part of the lane lines

From the binary image, the lane lines become clear with occasional noise.  To fit a polynomial and guess the middle of the lane lines are, we have to further refine which pixels are lanes and which (left or right) lane the pixels belong to. To make an initial guess of the lane positions, a histogram of the binary values along each column is calculated.  The histgram only evaluates pixels in the lower half of the image.

![alt text][hist]


The x axis represents columns in the combined binary image and the y axis is the number of (binary) pixels are 1 (and not 0) within that row. From looking at the peaks of the histogram and dividing along the center (pixel 640 in the horizontal direction), the initial position (lowest in the image) of both lanes are determined.  From here, we do a window search, shown below and explained after the image:

![alt text][win]

The windows heights are 40 pixels and widths are 150 pixels.  The first window is at the lowest part of the image and centered along the initial left and right values (determined from the histogram). All pixels within the lowest window are identified (and colored for visualization). The windows then shift up until they reach the highest pixels. With every shift up, the windows can be re-centered if more than 50 pixels are idntified in the window.  If so, the next window is centered around the mean of the column values of the current window. 

Once pixel values are identified, a second degree polynomial is fit along the values to determine the center of each lane (shown in yellow above).

Once this whole-image search is completed, in the next frame left and right lane pixels are identified if they simply lay within a margin (100 pixels to the left or 100 pixels to the right) of the polynomial fit. A new polynomial is then fit with these new pixel values.


### Radius of curvature

To calculate the radius of curvature, a conversion from pixels to meters is required. By assuming a standard lane width of 3.7 meters and counting horizontal pixels, it is determined that there are 3.7/672 pixels per meter in the horizontal direction.  By assuming a lane segment is 3 meters in the vertical direction, it is determined that there are 3/65 meters per pixel in the vertical direction.  By using these ratios to convert to meters and re-fitting the polynomial, the polynomial coefficients are used to calculat the radius of curvature for each lane in each frame, evaluated at the bottom of the frame. It was found that the measurement was quite sensitive and noisey and perhaps not reliable using this approach.

## Transforming back to the camera view and pipelining to overlay on a video

Before bringing the lane polynomial fits back to the image, some lane validation and smoothing steps are performed.  Every lane position is the average of the last 10 positions, which mediates jumping lane lines. There are also requirements to check that the lanes are roughly parallel and of an acceptable width. If these requirements are not met, the lane lines take on the those of the last good lane tracks.


Once lanes are verified, they warped back to the camera view and overlayed on top of the original image.  From here, assuming a lane width of 3.7 meters, we can also determine the position of the vehicle laterally within the lane.  Here, positive means to the right of the center.

![alt text][overlaid]

These steps are then placed in a pipeline and used to produce the above video

## Discussion

This approach doesn't require much data since there is no model to train but the manual tuning of parameters is tedious and only through many iterations and exposure to different driving conditions can a good set of parameters be found.  Even then there's no measurement of how well the parameters work on a wide array of videos. Tuning them on only one or two videos certainly won't be robust.

Improvement can be had at the binary transforms section by doing some weighted average not at the binary level but at the transformed level.  For example, each value of saturation will have a high confidence level based on closely within the parameters and if the same pixels are also within a good range with respect to the Sobel transforms then it can be of higher confidence.  The polynomial fit can then be weighted using the confidence of these pixels.

There was not testing for night time videos so that require more tuning or even a seperate set of parameters. In embedded systems, such added complexity will be weighed against limited device processing power, especially since the device will be busy with other tasks as well.  This approach seems to be computationally expensive, with the many transforms involved.
