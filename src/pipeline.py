from camera_calibration import create_camera
import pickle
import os
from debug import DEBUG
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get a calibrated camera and test it.
camera = create_camera()

def get_image_points():
    """
    The points on the source image that should represent a straight linear projection. I'm currently using
    this assuming the camera isn't distorted anymore.

    Note: we need to adjust for the angle of the camera and the offset within the car.
    """
    # Adjusts for where the horizon is on the screen. Given some slope of the camera the horizon will not be at
    # the center of the image. I calibrated this image on straight_line1.jpg.
    slope_factor = 0.82
    # How close to the horizon we should be looking at. If we go too close to the horizon tiny errors 
    # start to cause serious noise issues.
    interpolation_factor = 0.90
    # Factor sthat accounts for the position of the camera.
    top_left_fudge = 9
    top_right_fudge = 5

    distance_from_side = 190
    width = camera.viewport_size()[0]
    height = camera.viewport_size()[1]

    forty_width = int(width/2.0 * interpolation_factor + distance_from_side * (1-interpolation_factor))
    forty_height = int(height - height / 2.0 * interpolation_factor * slope_factor)
    image_points = [[distance_from_side, height], [forty_width - top_left_fudge, forty_height], [width - forty_width + top_right_fudge, forty_height], [width - distance_from_side, height]]
    return np.array(image_points, np.int32)

def get_destination_points():
    """
    Almost entire image shape, except for the extreme sides.
    """
    width = camera.viewport_size()[0]
    height = camera.viewport_size()[1]
    coords = [[200,height], [200,0], [width-200, 0], [width-200, height]]
    return np.array(coords, np.int32)

def bottom_half_histogram(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]
    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    return histogram

def mean_index(weights):
    indices = range(0, len(weights))
    return np.average(indices, weights=weights)

def find_lines(start_index, binary):
    """
    args:
        start_index: reasonable starting point. based off median in histogram
    """
    nwindows = 9 # Choose the number of sliding windows
    margin = 100 # Set the width of the windows +/- margin
    minpix = 40 # Set minimum number of pixels found to recenter window

    window_height = np.int(binary.shape[0]//nwindows)
    if binary.shape[0] % nwindows != 0:
        raise Exception("Image height not evenly divisible by number of windows.")
    current_center = start_index

    # Create empty lists to receive left and right lane pixel indices
    lane_inds = list()
    previous_lane_inds = list([current_center])

    for window_index in range(0, nwindows):
        # Start from the bottom up...
        window_index = nwindows -1 - window_index

        horizontal_slice = binary[window_index * window_height:(window_index+1) * window_height, :]
        window = horizontal_slice[:, current_center-margin:current_center+margin]
        window_f = np.array(window, np.float32)
        axis_sum = np.sum(window_f, axis=0)

        weights = axis_sum
        if np.sum(weights) >= minpix * 255:
            mean_local = mean_index(weights)
            mean = mean_local + current_center - margin
            current_center = int(mean)
            lane_inds.append(current_center)
        else:
            lane_inds.append(None)
        previous_lane_inds.append(current_center)
    output_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # draw start index
    index = nwindows - 0 - 1
    pts = [[start_index, window_height * index], [start_index, (index + 1) * window_height]]
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    output_image = cv2.polylines(output_image, [pts], True, (255,0,255), thickness = 4)

    output_path = []

    for index, lane_index in enumerate(lane_inds):
        # inverted index. Index from the bottom
        iindex = nwindows - index - 1
        
        # draw the detected center point, if we found one
        if lane_index is not None:
            pts = [[lane_index, window_height * iindex], [lane_index, (iindex + 1) * window_height]]
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1,1,2))
            output_image = cv2.polylines(output_image, [pts], True, (0,255,255), thickness = 4)
            output_path.append( [lane_index, window_height * iindex + window_height / 2.0] )

        # draw the box around where we looked
        lane_index = previous_lane_inds[index]
        pts = [[lane_index - margin, window_height * iindex], [lane_index + margin, window_height * iindex], [lane_index + margin, (iindex + 1) * window_height], [lane_index - margin, (iindex + 1) * window_height]]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        output_image = cv2.polylines(output_image, [pts], True, (0,255,255), thickness = 1)

    return output_image, output_path

    # TODO: create 
        #else:
        #print ("skipping. only " + str(np.sum(weights)) + " worth of pixels.")
        

def apply_filter(saturation_sobelx, grayscale_sobelx, start_height, end_height, binary_sobel, filter_amount, gray_filter_amount):
    saturation_window = saturation_sobelx[start_height:end_height,:]
    grayscale_window = grayscale_sobelx[start_height:end_height,:]
    binary_sobel[start_height:end_height,:][(saturation_window > filter_amount) | (saturation_window < -filter_amount) \
        | (grayscale_window > gray_filter_amount) | (grayscale_window < -gray_filter_amount)] = 255


def progressive_binary(saturation_warped, grayscale_warped, saturation_warped_threshold):
    # I think I'm supposed to be applying the threshold on the raw saturation value
    #raise Exception("I'm not certain I'm even supposed to be applying sobel on top of the saturation value.")

    # Sobel kernel of 3 for bottom half
    sobel_kernel = 3
    sobelx_kernel_3_s = cv2.Sobel(saturation_warped, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelx_kernel_3_gray = cv2.Sobel(grayscale_warped, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # right now, no matter what i do I can't find the top mark in the test2

    sobel_kernel = 5
    sobelx_kernel_5_s = cv2.Sobel(saturation_warped, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelx_kernel_5_gray = cv2.Sobel(grayscale_warped, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    DEBUG.save_img(grayscale_warped, "gray.png")
    DEBUG.save_img(sobelx_kernel_5_gray, "saturatsobelx_kernel_5_gray.png")
    DEBUG.save_img(sobelx_kernel_3_gray, "sobelx_kernel_3_gray.png")

    #sobel_kernel = 9
    #sobelx_kernel_9_s = cv2.Sobel(saturation_warped, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    #sobelx_kernel_9_gray = cv2.Sobel(grayscale_warped, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # TODO: don't just use the saturation binary. Also use the grayscale one. 
    # the two of them are better toghether.
    # https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/0bcd97c5-66f3-495b-9fe2-3f9f541bae25/concepts/a1b70df9-638b-46bb-8af0-12c43dcfd0b4

    # bottom_half = img[img.shape[0]//2:,:]
    height = saturation_warped.shape[0]

    binary_sobelx = np.zeros_like(saturation_warped)
    apply_filter(sobelx_kernel_3_s, sobelx_kernel_3_gray, height//2, height, binary_sobelx, 30, 50)
    apply_filter(sobelx_kernel_3_s, sobelx_kernel_5_gray, height//4, height//2, binary_sobelx, 20, 100)
    apply_filter(sobelx_kernel_3_s, sobelx_kernel_5_gray, 0, height//4, binary_sobelx, 20, 500)
    return binary_sobelx

def calculate_curvature_meters(pixel_path):
    """
    
    """
    # Define conversions in x and y from pixels space to meters
    # We were able to derive these by looking at the regulated lane width and 
    # dashed lane length.
    ym_per_pix = 20/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/850 # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose a value near the top of the screen.
    y_eval = np.max(600)

    path = [  [x[0] * xm_per_pix, x[1] * ym_per_pix] for x in pixel_path ]
    fit_y_to_x = np.polyfit([x[1] for x in path], [x[0] for x in path], 2)

    curvature = ((1 + (2*fit_y_to_x[0]*y_eval*ym_per_pix + fit_y_to_x[1])**2)**1.5) / np.absolute(2*fit_y_to_x[0])
    return curvature

def calculate_offset(left_lane, right_lane, screen_width):
    xm_per_pix = 3.7/850 # meters per pixel in x dimension

    center_of_screen = screen_width / 2.0
    center_of_lane = (left_lane + right_lane) / 2.0
    delta_pixels = center_of_lane - center_of_screen
    return delta_pixels * xm_per_pix

def process_frame(path, name):
    img = cv2.imread(path)
    return process_img(img, name)

def process_img(img, name):

    #
    # WARP TO NEW PERSPECTIVE
    #

    source_points = get_image_points()
    img = camera.undistort(img)
    DEBUG.save_img_with_path(img, source_points, name)

    source_points = np.array(source_points, np.float32)
    destination_points = np.array(get_destination_points(), np.float32)
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    warped = cv2.warpPerspective(img, M, camera.viewport_size(), flags=cv2.INTER_LINEAR)
    DEBUG.save_img(warped, "warped_" + name)

    #
    # CREATE BINARY IMAGE
    #

    # Create grayscale image
    grayscale_warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    hls_warped = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    saturation_warped = hls_warped[:,:,2]
    l_warped = hls_warped[:,:,1]
    #DEBUG.save_img(saturation_warped, "warped_saturation_" + name)
    saturation_warped_threshold = np.zeros_like(saturation_warped)

    # edge detection on a combination of saturation and grayscale.
    # saturation is mostly useful on yellow. So this is our best bet.
    binary_sobelx = progressive_binary(saturation_warped, grayscale_warped, saturation_warped_threshold)
    #DEBUG.save_img(binary_sobelx, "binary_" + name)

    # find the peaks in the bottom half of the image. This is the starting
    # point for finding the lane.
    histogram = bottom_half_histogram(binary_sobelx)

    max_first_half = histogram[:len(histogram)//2].argmax()
    max_second_half = histogram[len(histogram)//2:].argmax() + len(histogram)//2
    annotated_image, left_path = find_lines(max_first_half, binary_sobelx)
    #DEBUG.save_img(annotated_image, "annotated_output_left_" + name + ".png")
    annotated_image, right_path = find_lines(max_second_half, binary_sobelx)
    #DEBUG.save_img(annotated_image, "annotated_output_right_" + name + ".png")

    #
    # Fit polylines to the detected lane points
    # 

    # Generate x and y values for plotting in projected space
    fit_poly_left = np.polyfit([x[1] for x in left_path], [x[0] for x in left_path], 2)
    fit_poly_right = np.polyfit([x[1] for x in right_path], [x[0] for x in right_path], 2)
    ploty = np.linspace(0, binary_sobelx.shape[0]-1, binary_sobelx.shape[0])
    left_fitx = fit_poly_left[0]*ploty**2 + fit_poly_left[1]*ploty + fit_poly_left[2]
    right_fitx = fit_poly_right[0]*ploty**2 + fit_poly_right[1]*ploty + fit_poly_right[2]
    
    # Convert left path back to original space
    path_left = list(zip(left_fitx, ploty))
    path_left = np.array([(path_left)], dtype=np.float32)
    reverse_transform = cv2.getPerspectiveTransform(destination_points, source_points)
    converted_left_path = cv2.perspectiveTransform(path_left, reverse_transform)
    
    # Convert right path back to original space
    path_right = list(zip(right_fitx, ploty))
    path_right = np.array([(path_right)], dtype=np.float32)
    converted_right_path = cv2.perspectiveTransform(path_right, reverse_transform)

    # Draw bounding box around the lane area
    overlay_image = np.zeros_like(img)
    bounding_array = np.concatenate( (converted_left_path[0], np.flipud(converted_right_path[0])) )
    bounding_box = np.array([bounding_array], dtype=np.int32)
    overlay_image = cv2.fillPoly(overlay_image, bounding_box, (0,255, 0))

    # this works fine
    #DEBUG.save_img(overlay_image, "overlay_image_" + name + ".png")
    result_image = cv2.addWeighted(img, 1, overlay_image, 0.3, 0)
    #DEBUG.save_img(result_image, "zoutput_" + name + ".png")

    # use plot to get us an image
    path_left = list(zip(left_fitx, ploty))
    path_left = np.array([(path_left)], dtype=np.float32)
    converted_left_path = cv2.perspectiveTransform(path_left, reverse_transform)
    path_right = list(zip(right_fitx, ploty))
    path_right = np.array([(path_right)], dtype=np.float32)
    converted_right_path = cv2.perspectiveTransform(path_right, reverse_transform)
    
    #
    # Curvature & offset in lane
    #

    # Calculate values
    left_curvature = calculate_curvature_meters(left_path)
    right_curvature = calculate_curvature_meters(right_path)
    average_curvature = (left_curvature + right_curvature) / 2.0
    offset = calculate_offset(left_path[0][0], right_path[0][0], binary_sobelx.shape[1])

    # Annotate the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Radius of curvature = {}m'.format(int(average_curvature))
    text2 = 'Offset from center = {}m'.format(offset)
    result_image = cv2.putText(result_image, text, (100,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    result_image = cv2.putText(result_image, text2, (100,150), font, 1,(255,255,255),2,cv2.LINE_AA)
    DEBUG.save_img(result_image, "zzzfinal_output_" + name + ".png")
    return result_image
  
