#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  
# Your goal is:
# -- piece together a pipeline to detect the line segments in the image, 
# -- then average/extrapolate them and 
# -- draw them onto the image for display (as below). 
# -- Once you have a working pipeline, try it out on the video stream below.**
# 
# ---

### some functions are adapted from  https://github.com/naokishibuya/car-finding-lane-lines

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# color and region selection criteria
red_threshold   = 150
green_threshold = 150
blue_threshold  = 150
apex_y          = 305
bottom_x_left   = 450
bottom_x_right  = 490
# Gaussian smoothing/blurring paramteres
kernel_size = 5 # Must be an odd number (3, 5, 7...)
# Canny paramteres
canny_low_threshold = 50
canny_high_threshold = 150
# Hough Transform parameters
rho             = 1         # distance resolution in pixels of the Hough grid
theta           = np.pi/180 # angular resolution in radians of the Hough grid
threshold       = 55        # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40        # minimum number of pixels making up a line
max_line_gap    = 250       # maximum gap in pixels between connectable line segments

# ## Helper Functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    #return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
    return weighted_img(line_image, image, 1.0, 0.95, 0.0)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def plot_debug(image, name):
    print(name)
    fig = plt.gcf()
    plt.imshow(image)
    plt.show()
    fig.savefig('examples' + "/" + name + '.png')
   
def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    return cv2.bitwise_and(image, image, mask = mask)


def select_white(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)

    mask = white_mask    
    return cv2.bitwise_and(image, image, mask = mask)


   
####################################################################################
#### Get lines for left lane and right lane
###################################################################################
def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights, left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))


def get_lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

####################################################################################
#### Edge detection -- gray, hough and canny
###################################################################################
def get_canny_edges(image):
    gray_image = grayscale(image)
    blur_gray_image = gaussian_blur(gray_image, kernel_size)
    edges = canny(blur_gray_image, canny_low_threshold, canny_high_threshold)
    
    return edges

########################################################################
### Region of Interest Selection -- defining a four sided polygon to mask
########################################################################
def get_region_edges(edges, image):
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(bottom_x_left, apex_y), 
                          (bottom_x_right, apex_y), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    return masked_edges

########################################################################
### Hough Transform Line Detection
########################################################################
def get_hough_lines(masked_edges):
    # Run Hough on masked edges
    lines = hough_lines(masked_edges, rho, theta, threshold, 
                        min_line_length, max_line_gap)    
    # Output "lines" is an array containing endpoints of detected line segments
    return lines


########################################################################
### Process Image
########################################################################
def process_image(image):
    edges = get_canny_edges(image)
    masked_edges = get_region_edges(edges, image)
    lines = get_hough_lines(masked_edges)
    lane_lines =  get_lane_lines(image, lines)
    result_image = draw_lane_lines(image, lane_lines)
    #plot_debug(result_image, "result_image")
    
    return result_image

def process_image_wy(image):
    edges = get_canny_edges(select_white_yellow(image))
    masked_edges = get_region_edges(edges, image)
    lines = get_hough_lines(masked_edges)
    lane_lines = get_lane_lines(image, lines)
    result_image = draw_lane_lines(image, lane_lines)
    #plot_debug(result_image, "result_image")
    
    return result_image

########################################################################
### Process Videos
########################################################################
def process_video(video_input, video_output, process_image_fn):
    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(process_image_fn)
    processed.write_videofile(os.path.join('test_videos_output', video_output), audio=False)


def main():
    #os.listdir("test_images/")
    #image = mpimg.imread('test_images/solidWhiteRight.jpg')
    #process_image(image)
    
    process_video('solidWhiteRight.mp4', 'solidWhiteRight_output.mp4', process_image)
    process_video('solidYellowLeft.mp4', 'solidYellowLeft_output.mp4', process_image)
    process_video('challenge.mp4', 'challenge_output.mp4', process_image_wy)
    

main()
