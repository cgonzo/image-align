# read_chip.py
# Chris Gonzales
#
# Reads brightness from points on DNA chip
# Takes an image of a chip, aligns it to a reference
# And then reads brightness from pre-defined spots

import cv2
import argparse
import numpy as np

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Image to be aligned", required=True)
ap.add_argument("-o", "--output", help = "Output csv file name", required=True)
ap.add_argument("-a", "--alignmentblur", type = int, default=7, 
	help = "Radius of Gaussian blur for alignment; must be odd.\nMore blur allows for better noise filtering but worse alignment. Default is 7.")
ap.add_argument("-d", "--dotsize", type = int, default=6, 
	help = "Radius of alignment dots.\n This is used to mask off already-found alignment dots and calculate average circl brightness ")
ap.add_argument("-n", "--numdots", type = int, default=12, 
	help = "Number of alignment dots")
ap.add_argument("-s", "--showalignment", help = "Display aligned images on screen", action="store_true")
ap.add_argument("-t", "--troubleshooting", help = "Troubleshooting mode", action="store_true")
args = vars(ap.parse_args())

# Read the images to be aligned
im1 =  cv2.imread(args["image"])
alignmentblur = args["alignmentblur"]
dotsize = args["dotsize"]
numdots = args["numdots"]
output_file = open(args["output"],'w')

print "Image size is %d x %d" %(im1.shape[1],im1.shape[0])

# Convert image to grayscale
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
# Apply a blur to filter out any noise.
im1_alignmentblur = cv2.GaussianBlur(im1_gray,(alignmentblur,alignmentblur),0)

# Search for alignment dots
# We're repeatedly finding the maximum brightness point, then masking the area of it off
# then finding the next-brightest, until we have all dots accounted for.
im1_dots=np.empty([numdots,2])
mask1 = im1_gray.copy()
mask1[:]=255
for i in xrange(numdots):
  (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(im1_alignmentblur,mask1)
  im1_dots[i]=maxLoc
  cv2.circle(mask1, maxLoc, dotsize*2, 0, -1)
# sort im1_dots first by y, then separately sort the upper and lower 6
im1_dots_sorted = im1_dots[im1_dots[:,1].argsort()]
im1_upper = im1_dots_sorted[6:]
im1_upper_sorted = im1_upper[im1_upper[:,0].argsort()]
im1_lower = im1_dots_sorted[0:6]
im1_lower_sorted = im1_lower[im1_lower[:,0].argsort()]
im1_dots_sorted = np.concatenate((im1_lower_sorted, im1_upper_sorted))

# Right now, all the sample images are 1024x1280, so only support that resolution
# Would be easy enough to scale either the images or sampling matrix to a new resolution
# The chip is 17x18. 
# From the layout.xls file:
# Alignment dots in the matrix are in the first row at 1,4,7,10,13,18
# In the 17th row at 1,4,7,14,16,18
ref_alignment_dots_18x17 = np.array([[18,1],
                                     [13,1],
                                     [10,1],
                                     [7,1],
                                     [4,1],
                                     [1,1],
                                     [18,17],
                                     [16,17],
                                     [14,17],
                                     [7,17],
                                     [4,17],
                                     [1,17]])

# First of all, the images are mirrored on the X-axis, so need to subtract X from 18
ref_alignment_dots_mirrored = ref_alignment_dots_18x17.copy()
ref_alignment_dots_mirrored[:,0] = 18 - ref_alignment_dots_18x17[:,0]
# The chip is 17x18. To make this easier, we'll use center points in a 32x32 matrix
# This means paddings of 7 on the left right and top, 8 on bottom
ref_padding=7
ref_alignment_dots_32x32 = ref_alignment_dots_mirrored + ref_padding

#Now to scale this to a height of 1024, multiply by 32
ref_scaling=32
ref_alignment_dots_1280x1024 = ref_alignment_dots_32x32 * ref_scaling


# Calculate homography using the image and alignment dots. 
# We're using RANSAC since it seems to give the best answer
warp_matrix, status = cv2.findHomography(im1_dots_sorted, ref_alignment_dots_1280x1024, cv2.RANSAC)
percent_inliers = np.sum(status)/numdots
print "Percentage of inliers: %f" %(percent_inliers*100)

# Then line up im1 with alignment dots
sz = im1.shape
im1_aligned = cv2.warpPerspective (im1, warp_matrix, (sz[1],sz[0]))
im1_aligned_gray = cv2.cvtColor(im1_aligned,cv2.COLOR_BGR2GRAY)

# Now measure all the points we care about
dots = im1_aligned.copy()
brightness = np.zeros((20,10))
# We actually use the areas to the left and right of the array as references, so x is -1..19 rather than 1..18
# for y it's even stranger; we only care about rows 7-16 (not the top 5 rows)
print >>output_file, "Row,Column,Brightness"
for y in range(7,17):
  measurement_point_y = (y + ref_padding) * ref_scaling
  for x in range(-1,19):
    measurement_point_x = (x + ref_padding) * ref_scaling
    for x_i in range(measurement_point_x - dotsize,measurement_point_x + dotsize):
      for y_i in range(measurement_point_y - dotsize,measurement_point_y + dotsize):
        brightness[x+1,y-7] += im1_aligned_gray[y_i,x_i]
    print >>output_file, "%d,%d,%d" %(y-7, x+1, brightness[x+1,y-7])
    cv2.circle(dots, (measurement_point_x,measurement_point_y), dotsize*2, brightness[x+1,y-7]/10, -1)
cv2.imwrite("dots.png",dots)
cv2.imshow("dots",dots)
cv2.waitKey(0)
output_file.close()

