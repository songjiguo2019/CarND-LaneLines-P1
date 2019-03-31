# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image_edges]: ./examples/edges.png "Edges"
[image_masked_edges]: ./examples/masked\_edges.png "Masked_Edges"
[image_result]: ./examples/result\_image.png "Result"

---

### Reflection

**Important Clarification**: all code is in **P1.py** and the report file is **my_writeup.md**

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

At the highest level, the project logically consists of 2 parts:
single image processing and video processing (i.e., continuous image
processing). The main pipeline for single image processing consisted
of 5 steps, as followings:

* **First step** -- get edges by applying Canny Edge Detection onto a
 copy of original image that has been processed by grayscale and
 Gaussian Smoothing method (i.e., blurring with kernel size 5). For
 Canny Edge Detection, we set the low threshold to be 50 and high
 threshold to be 150 respectively.

	* *Additional function*: when one of the lane is painted in
      yellow, we preprocess the image with *select_white_yellow*
      function to filter out everything but white and yellow. By doing
      this, we can preserve most RGB information of both white and
      yellow lanes.

	```

	# Gaussian smoothing/blurring paramteres
	kernel_size = 5
	# Canny paramteres
	canny_low_threshold = 50
	canny_high_threshold = 150
	
	def get_canny_edges(image):
    gray_image = grayscale(image)
    blur_gray_image = gaussian_blur(gray_image, kernel_size)
    edges = canny(blur_gray_image, canny_low_threshold, canny_high_threshold)
    
    return edges

	```
![alt text][image_edges] 


* **Second step** -- selected edges in a predetermined region, which
 usually is defined by the view angle of camera that is mounted on the
 top of vehicle. All it does is to define a geometry region and call
 the provided function *region_of_interest*. Based on the image, we
 set the region using following parameters:
 
	```
 
	apex_y          = 305
	bottom_x_left   = 450
	bottom_x_right  = 490
 
	def get_region_edges(edges, image):
	imshape = image.shape
	vertices = np.array([[(0,imshape[0]),(bottom_x_left, apex_y), 
		(bottom_x_right, apex_y), (imshape[1],imshape[0])]], dtype=np.int32)
	
	masked_edges = region_of_interest(edges, vertices)
 
    return masked_edges
	
	```

![alt text][image_masked_edges]

* **Third step** -- apply Hugh Transform Line Detection algorithm on
  the selected edges from previous step to get an array of detected
  line segments.
  
    ```
	def get_hough_lines(masked_edges):
		lines = hough_lines(masked_edges, rho, theta, threshold, 
			                 min_line_length, max_line_gap)

		return lines

	```


* **Fourth step** -- Convert line segments we have got into their
  pixel points representation from their slope and intercept.

	```
	def get_lane_lines(image, lines):
		left_lane, right_lane = average_slope_intercept(lines)
    
		y1 = image.shape[0] # bottom of the image
		y2 = y1*0.6         # slightly lower than the middle

		left_line  = make_line_points(y1, y2, left_lane)
		right_line = make_line_points(y1, y2, right_lane)
    
		return left_line, right_line
	```
	
	In order to draw a single line on the left and right lanes, we
	also averaged all line segment's slope and intercept according to
	their being negative or positive. For left lane, line's slop
	should be less than 0 and it should be greater than 0 for right
	lane. Then the average of slope and intercept is computed by
	dividing their sum product with length by the total line
	length. For more details, please refer to following function
	
	```
	
	def average_slope_intercept(lines)

	```
	
	
* **Last step** -- Overlap those pixel-point lane lines to the
  original image. **Keep in mind, those lines have been processed in
  previous step and they are representing averaged slop and intercept
  for the single lane already -- another way to say, the change in
  original draw_lines function has been done in get_lane_lines
  function as we explained above.**
  
  ```
  def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
		
	return weighted_img(line_image, image, 1.0, 0.95, 0.0)
  
  ```

![alt text][image_result]


* **Video process** -- The pipeline for video processing is basically
  same as the image processing, except it extracts each frame as an
  image and apply the image processing pipeline. Eventually save the
  output into another video with lanes being marked. For all video,
  you can find them under test\_videos\_output.
  
  - solidYelloLeft_output.mp4
  - solidWhiteRight_output.mp4
  - challenge_output.mp4 (not perfect)


### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when the lane
color is not white nor yellow. Another potential shortcoming would be
when lanes have some sharp curve, up/down hill environment or even
being covered by obstacles (i.e., other vehicle that is crossing the
lane we are detecting), which might break our assumption about region
and the threshold used by Canny Edge Detection or Hough Transform
algorithms.

Another shortcoming could be the changes in ambient light environment
(e.g., daytime vs nighttime, bad weather vs good weather, large area
shadow etc.). This could affect the color assumption we have made.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to allow dynamically input those
above-mentioned parameters, according to the environment change or by
automatic detection. Maybe one way to do this is to calculate the
inputs by combining with pre-calculated map.

Another potential improvement could be to use different type camera
that can handle the environment well (i.e., night vision or
infrared). However, the most part of skills we have learned in this
project should stay unchanged.
