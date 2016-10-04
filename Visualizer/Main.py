import logging
import os

import colour
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread

from TimeTravelTaskBinaryReader import read_binary_file, get_filename_meta_data

local_directory = os.path.dirname(os.path.realpath(__file__))  # The directory of this script
filename = '001_1_1_1_2016-08-29_10-26-03.dat'  # The relative path to the data file (CHANGE ME)
meta = get_filename_meta_data(filename)  # The meta filename information for convenience
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

# First we populate a list of each iteration's data
# This section of code contains some custom binary parser data which won't be explained here
iterations = read_binary_file(os.path.join(local_directory, filename))

# Determine the background image according to meta phase
bg_path = 'studyBG.png'
if meta['phase'] == '0':
    bg_path = 'practiceBG.png'
img = imread(os.path.join(local_directory, bg_path))

# Generate a list of colors for the lines based on time (index = seconds in time)
yellow = colour.Color(color="yellow")
red = colour.Color(color="red")
green = colour.Color(color="green")
blue = colour.Color(color="blue")
colors = []
colors.extend([yellow] * 15)
colors.extend([red] * 15)
colors.extend([green] * 15)
colors.extend([blue] * 16)

# Create lists for visualization data
segs = []
cols = []
buttonEventsX = []
buttonEventsY = []

# Determine if the timeline should be subsampled because we're in practice (which is have the length)
color_subsample_factor = 1
if meta['phase'] == '0':
    color_subsample_factor = 2

# Generate the visualization data for each iteration
for i in range(0, len(iterations)-2):
    # Get the x, y pairs for each line segment
    x1 = iterations[i]['x']
    x2 = iterations[i+1]['x']
    y1 = iterations[i]['z']
    y2 = iterations[i+1]['z']
    # Compute the color according to the time and phase
    c = colors[int(iterations[i]['time']/color_subsample_factor)].rgb
    # Add line segment and color to lists
    segs.append(((x1, y1), (x2, y2)))
    cols.append(c)
    # Determine if any button other than the first (X/time travel button) are pressed and add point to list of true
    if any(iterations[i]['buttons']):
        buttonEventsX.append(x1)
        buttonEventsY.append(y1)
# Check the last iteration as it is ignored by the line segmentation
if any(iterations[-1]['buttons'][1:]):
    x1 = iterations[-1]['x']
    y1 = iterations[-1]['z']
    buttonEventsX.append(x1)
    buttonEventsY.append(y1)

# Output the iterations count for debugging purposes
logging.info("Plotting " + str(len(segs)+1) + " iterations.")

# Render the background image with the appropriate coordinate system
# (the extents are 19 because the wall is 1 unit thick)
plt.imshow(img, zorder=0, extent=[-19, 19, -19, 19])

# Generate a line collection visualization and render it
ln_coll = matplotlib.collections.LineCollection(segs, colors=cols)
ax = plt.gca()
ax.add_collection(ln_coll)

# Render the scatter plot of button events
plt.scatter(buttonEventsX, buttonEventsY, s=5, c=(0, 0, 0))

logging.info("Showing plot. Close plot to exit program.")
# Show the plot
plt.show()

logging.info("Done")
