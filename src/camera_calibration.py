import os
import os.path
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing

from debug import DEBUG


# what is my first task? Probably to examine the chess board images.
# You need to set your chessboard size to 9x6 for the project

# if I'm trying to calibrate the camera, don't i need ot know where the actual chess baord is?
# maybe just need to understand the objectPoints. Ie, the relative location within the object.

def create_camera():
    if os.path.exists("camera_cache.pickle"):
        with open("camera_cache.pickle", "rb") as f:
            return pickle.load(f)
    else:
        camera = Camera()
        with open("camera_cache.pickle", "wb") as f:
            pickle.dump(camera, f)
        return camera

class Camera:
    def __init__(self):
        imgpoints, objpoints = image_points_and_object_points()
        self.shape = image_shape()
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                objectPoints=objpoints, 
                imagePoints=imgpoints, 
                imageSize=self.shape,
                cameraMatrix=None,
                distCoeffs=None)
        if not ret:
            raise Exception("Calibration failed.")

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def viewport_size(self):
        return self.shape

    def debug_test_img(self):
        img = cv2.imread("./camera_cal/calibration1.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.undistort(gray)
        DEBUG.save_img(gray, "test_undistorted_calibration0.jpg")

        img = cv2.imread("./test_images/test1.jpg")
        img = self.undistort(img)
        DEBUG.save_img(img, "test_undistorted_test1.jpg")

vertial = 6
horizontal = 9

def object_points():
    """
    Returns the coordinate points of all the corners in a chess board.
    This is constant for all images since the coordinates are in the "object space".
    The coordinates are always relative the object itself.
    """
    object_points = np.zeros( (horizontal*vertial, 3), np.float32)
    object_points[:,:2] = np.mgrid[0:horizontal,0:vertial].T.reshape(-1,2)
    return object_points

def image_shape():
    """

    """
    img = cv2.imread("./camera_cal/calibration1.jpg")
    return img.shape[::-1][1:3]

def load_chessboard_images():
    camera_dir = "./camera_cal/"
    for path in os.listdir(camera_dir):
        img = cv2.imread(camera_dir + path)
        yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def process_one_calibration_image(input):
    index, img_gray = input
    ret, corners = cv2.findChessboardCorners(img_gray, (horizontal,vertial), None)
    if ret:
        # Uncomment to debug.
        # debug_img = cv2.drawChessboardCorners(img_gray, (horizontal,vertial), corners, ret)
        # DEBUG.save_img(debug_img, "corners" + str(index) + ".jpg")
        return (corners, object_points())
    else:
        print ("Skipping calibration image " + str(index) + " since corners couldn't be found.")

def image_points_and_object_points():
    imgpoints = list()
    objpoints = list()
    pool = multiprocessing.Pool(processes=4)
    results = pool.map(process_one_calibration_image, enumerate(load_chessboard_images()))
    imgpoints = [result[0] for result in results if result is not None]
    objpoints = [result[1] for result in results if result is not None]
    return imgpoints, objpoints

