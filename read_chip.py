# read_chip.py
# Chris Gonzales
#
# Reads brightness from points on DNA chip
# Takes an image of a chip, aligns it to a reference
# And then reads brightness from pre-defined spots

import cv2
import argparse
import numpy as np
import json
from pprint import pprint

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Image to be aligned", required=True)
ap.add_argument("-o", "--output", help = "Output csv file name", required=True)
ap.add_argument("-c", "--config", help = "Input JSON config file", required=True)
ap.add_argument("-a", "--alignmentblur", type = int, default=7, 
	help = "Radius of Gaussian blur for alignment; must be odd.\nMore blur allows for better noise filtering but worse alignment. Default is 7.")
ap.add_argument("-d", "--dotsize", type = int, default=6, 
	help = "Radius of alignment dots.\n This is used to mask off already-found alignment dots and calculate average circl brightness ")
ap.add_argument("-s", "--showalignment", help = "Display aligned images on screen", action="store_true")
ap.add_argument("-t", "--troubleshooting", help = "Troubleshooting mode", action="store_true")
args = vars(ap.parse_args())

# Store arguments in pyton variables
alignmentblur = args["alignmentblur"]
dotsize = args["dotsize"]
output_file = open(args["output"],'w')
config_file = open(args["config"],'r')
im1 =  cv2.imread(args["image"])
if args["troubleshooting"]:
  print "alignmentblur: %f" %(alignmentblur)
  print "dotsize: %f" %(dotsize)

# Print info about the image we are given
xsize = im1.shape[1]
ysize = im1.shape[0]
print "Image size is %d x %d" %(xsize,ysize)

# Parse json config file
config_contents = json.load(config_file)
if args["troubleshooting"]:
  pprint(config_contents)
# Amount of padding from left when building reference image from reference dots
padding=config_contents["padding"]
# Scaling factor for reference dots. Right now, scaling X and Y by same factor
scaling=config_contents["scaling"]
# Get the locations of the dots we actually want to read. Note that these can be negative if we want to read to the left or top of a reference dot
min_y=config_contents["min_y"]
max_y=config_contents["max_y"]
min_x=config_contents["min_x"]
max_x=config_contents["max_x"]
assert (min_x + padding) * scaling > 0 , "sample dots less than 0 once scaled in x direction. Increase padding or scaling"
assert (min_y + padding) * scaling > 0 , "sample dots less than 0 once scaled in y direction. Increase padding or scaling"
assert (max_x + padding) * scaling < xsize, "sample dots out of bounds once scaled in x direction. Reduce padding or scaling"
assert (max_y + padding) * scaling < ysize, "sample dots out of bounds once scaled in y direction. Reduce padding or scaling"
# Config file has list containing location of reference dots. Need to convert to numpy array for cv2 to use it
ref_alignment_dots_raw = np.array(config_contents["ref_dots"])
# Number of dots is simply the number of (x,y) tuples in this array
numdots = ref_alignment_dots_raw.shape[0]
if args["troubleshooting"]:
  print "Number of alignment dots: %f" %(numdots)
(ref_max_x,ref_max_y) = np.amax(ref_alignment_dots_raw,axis=0)
(ref_min_x,ref_min_y) = np.amin(ref_alignment_dots_raw,axis=0)
# Now we need to sort the reference dots. We want to sort by y descending and then x ascending
ref_alignment_dots_order = np.lexsort((-ref_alignment_dots_raw[:,0],ref_alignment_dots_raw[:,1]))
ref_alignment_dots_sorted = ref_alignment_dots_raw[ref_alignment_dots_order]
pprint(ref_alignment_dots_sorted)
#  Make sure that scaling makes sense
assert (ref_min_x + padding) * scaling > 0 , "reference dots less than 0 once scaled in x direction. Increase padding or scaling"
assert (ref_min_y + padding) * scaling > 0 , "reference dots less than 0 once scaled in y direction. Increase padding or scaling"
assert (ref_max_x + padding) * scaling < xsize, "reference dots out of bounds once scaled in x direction. Reduce padding or scaling"
assert (ref_max_y + padding) * scaling < ysize, "reference dots out of bounds once scaled in y direction. Reduce padding or scaling"
# Add padding to the left and top. Right now, we use the same padding for both
ref_alignment_dots_padded = ref_alignment_dots_sorted + padding
# Now scale this to the size of the image
ref_alignment_dots_scaled = ref_alignment_dots_padded * scaling

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
# sort im1_dots
#  We know from the reference dots how these are supposed to be lined up
#  So what we do is sort the image's dots by y, then use the reference dots to know how many are supposed
#  to be in each row. We bin them into these rows and then sort by x.
im1_dots_sorted_by_y = im1_dots[im1_dots[:,1].argsort()]
ref_alignment_dots_bins,ref_alignment_dots_bins_counts = np.unique(ref_alignment_dots_scaled[:,1],return_counts=True)
pprint(ref_alignment_dots_bins_counts)
lower_limit=0
im1_dots_sorted=np.empty(shape=[0,2])
for i in range(0,ref_alignment_dots_bins_counts.size):
  print i
  upper_limit = lower_limit + ref_alignment_dots_bins_counts[i]
  im1_subarray = im1_dots_sorted_by_y[lower_limit:upper_limit]
  im1_subarray_sorted = im1_subarray[im1_subarray[:,0].argsort()]
  pprint(im1_subarray_sorted)
  im1_dots_sorted = np.concatenate((im1_dots_sorted, im1_subarray_sorted))
  lower_limit = upper_limit
if args["troubleshooting"]:
  print "Sorted reference dot array:"
  pprint(im1_dots_sorted)

# Calculate homography using the image and alignment dots. 
# We're using RANSAC since it seems to give the best answer
warp_matrix, status = cv2.findHomography(im1_dots_sorted, ref_alignment_dots_scaled, cv2.RANSAC)
percent_inliers = float(np.sum(status))/float(numdots)
if args["troubleshooting"]:
  print "findHomography status array:"
  pprint(status)
  print "Sum of inliers: %f" %(np.sum(status))
  print "percent_inliers: %f" %(percent_inliers)
print "Percentage of inliers: %f" %(percent_inliers*100)

# Then line up im1 with alignment dots
im1_aligned = cv2.warpPerspective (im1, warp_matrix, (xsize,ysize))
im1_aligned_gray = cv2.cvtColor(im1_aligned,cv2.COLOR_BGR2GRAY)

# Now measure all the points we care about
dots = im1_aligned.copy()
brightness = np.zeros((max_x-min_x,max_y-min_y))
# We actually use the areas to the left and right of the array as references, so x is -1..19 rather than 1..18
# for y it's even stranger; we only care about rows 7-16 (not the top 5 rows)
print >>output_file, "Row,Column,Brightness"
for y in range(min_y,max_y+1):
  measurement_point_y = (y + padding) * scaling
  for x in range(min_x,max_x+1):
    measurement_point_x = (x + padding) * scaling
    for x_i in range(measurement_point_x - dotsize,measurement_point_x + dotsize):
      for y_i in range(measurement_point_y - dotsize,measurement_point_y + dotsize):
        brightness[x-min_x,y-min_y] += im1_aligned_gray[y_i,x_i]
    print >>output_file, "%d,%d,%d" %(y-min_y, x-min_x, brightness[x-min_x,y-min_y])
    cv2.circle(dots, (measurement_point_x,measurement_point_y), dotsize*2, brightness[x-min_x,y-min_y]/10, -1)
cv2.imwrite("dots.png",dots)
cv2.imshow("dots",dots)
cv2.waitKey(0)
output_file.close()

