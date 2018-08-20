import os
import cv2

class Debug(object):
    def __init__(self):
        os.system("rm -rf ./debug_images/")
        os.system("mkdir ./debug_images/")

    def isEnabled():
        return False

    def save_img(self, img, name):
        return None #No nop
        cv2.imwrite("./debug_images/" + name, img)

    def save_img_with_path(self, img, path, name):
        return None #No nop
        pts = path.reshape((-1,1,2))
        img = img.copy()
        img = cv2.polylines(img, [pts], True, (0,255,255))
        self.save_img(img, name)

DEBUG = Debug()