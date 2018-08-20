import cv2
import pipeline
import os

def create_video(dir_path):
  images = []
  for f in os.listdir(dir_path):
      if f.endswith("png"):
          images.append(f)

  # Determine the width and height from the first image
  image_path = os.path.join(dir_path, images[0])
  frame = cv2.imread(image_path)
  cv2.imshow('video',frame)
  height, width, channels = frame.shape

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
  out = cv2.VideoWriter("output.mp4v", fourcc, 20.0, (width, height))

  for image in images:
      image_path = os.path.join(dir_path, image)
      frame = cv2.imread(image_path)
      out.write(frame) # Write out frame to video
      cv2.imshow('video',frame)
      if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
          break

  # Release everything if job is finished
  out.release()
  cv2.destroyAllWindows()

def process_frames(dir_name):
  vidcap = cv2.VideoCapture('project_video.mp4')
  success, image = vidcap.read()
  count = 0
  while success:
    name = str(count).zfill(4) 
    processed_image = pipeline.process_img(image, name)
    cv2.imwrite(dir_name + name + ".png", processed_image)
    count = count + 1
    success, image = vidcap.read()

os.system("mkdir temp_frame_output")
process_frames("temp_frame_output/")
create_video("temp_frame_output/")
os.system('rm -rf temp_frame_output')