
import glob
import cv2

def extract_hog_features(images):
  hog = cv2.HOGDescriptor()
  hog_features = []
  for image in images:
    hog_features.append(hog.compute(image))
  return hog_features

# List all the images in the folder
image_files = glob.glob("C:/Users/Hamza/Desktop/test/aaaa/*.jpg")

# Load the images
images = []
for file in image_files:
  image = cv2.imread(file)
  images.append(image)

# Extract HOG features for the images
hog_features = extract_hog_features(images)






