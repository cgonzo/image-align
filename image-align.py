# image_align.py
# Chris Gonzales
#
# Finds alignment dots in one image and transforms
# that image to match the alignment of a reference image

import cv2
import argparse
import numpy as np

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--match", help = "Image to be matched", required=True)
ap.add_argument("-a", "--align", help = "Image to be aligned", required=True)
ap.add_argument("-o", "--output", help = "Output image file name")
ap.add_argument("-b", "--blur", type = int, default=7, 
	help = "Radius of Gaussian blur; must be odd.\nMore blur allows for better noise filtering but worse alignment. Default is 7.")
ap.add_argument("-d", "--dotsize", type = int, default=10, 
	help = "Maximum size of alignment dots.\n This is used to mask off already-found alignment dots ")
ap.add_argument("-n", "--numdots", type = int, default=12, 
	help = "Number of alignment dots")
ap.add_argument("-s", "--showalignment", help = "Display aligned images on screen", action="store_true")
ap.add_argument("-t", "--troubleshooting", help = "Troubleshooting mode", action="store_true")
args = vars(ap.parse_args())

# Read the images to be aligned
im1 =  cv2.imread(args["match"])
im2 =  cv2.imread(args["align"])
blur = args["blur"]
dotsize = args["dotsize"]
numdots = args["numdots"]

print "Image 1 size is %d x %d" %(im1.shape[:2])
print "Image 2 size is %d x %d" %(im2.shape[:2])
 
# Convert images to grayscale
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
# Apply a blur to filter out any noise.
im1_blur = cv2.GaussianBlur(im1_gray,(blur,blur),0)
im2_blur = cv2.GaussianBlur(im2_gray,(blur,blur),0)

# Search for alignment dots
# We're repeatedly finding the maximum brightness point, then masking the area of it off
# then finding the next-brightest, until we have all dots accounted for.
im1_dots=np.empty([numdots,2])
mask1 = im1_gray.copy()
mask1[:]=255
for i in xrange(numdots):
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(im1_blur,mask1)
  im1_dots[i]=maxLoc
  cv2.circle(mask1, maxLoc, dotsize, 0, -1)
# sort im1_dots from http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
im1_dots.view('i8,i8').sort(axis=0)

im2_dots=np.empty([numdots,2])
mask2 = im2_gray.copy()
mask2[:]=255
for i in xrange(numdots):
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(im2_blur,mask2)
  im2_dots[i]=maxLoc
  cv2.circle(mask2, maxLoc, 10, 0, -1)
im2_dots.view('i8,i8').sort(axis=0)

# Calculate homography using the dots. We're using RANSAC since it seems to give the best answer
warp_matrix, status = cv2.findHomography(im2_dots, im1_dots, cv2.RANSAC)
#warp_matrix, status = cv2.findHomography(im2_dots, im1_dots, cv2.LMEDS)
percent_inliers = np.sum(status)/numdots
print "Percentage of inliers: %f" %(percent_inliers*100)

# Then line up im2 with im1
sz = im1.shape
im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]))

# Output final image
if (percent_inliers > 0.5):
  if args["output"]:
    cv2.imwrite(args["output"],im2_aligned)
else:
  print "ERROR: More than 50% outliers. Cannot match images"

# For troubleshooting, you can show the masks and make sure the dots are where you expect
if args["troubleshooting"]:
  print im1_dots
  print im2_dots
  print warp_matrix
  print status
  cv2.imshow("Mask1", mask1)
  cv2.imshow("Mask2", mask2)
  cv2.waitKey(0)
 
# Show final results
if args["showalignment"]:
  # additional troubleshooting step, show the alignment of the alignment dots
  mask2_aligned = cv2.warpPerspective (mask2, warp_matrix, (sz[1],sz[0]))
  cv2.imshow("Original images overlaid", cv2.addWeighted(im1,0.5,im2,0.5,0))
  cv2.imshow("Masks overlaid", cv2.addWeighted(mask1,0.5,mask2_aligned,0.5,0))
  cv2.imshow("Aligned images overlaid", cv2.addWeighted(im1,0.5,im2_aligned,0.5,0))
  cv2.waitKey(0)

