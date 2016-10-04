import argparse
import logging
import os
import tkFileDialog

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import tkinter
from pyqtgraph.Qt import QtCore, QtGui
from scipy.misc import imread

from TimeTravelTaskBinaryReader import read_binary_file, get_filename_meta_data

########################################################################################################################
# Parse Arguments
########################################################################################################################

# Test args with
# --log_file "C:\Users\Kevin\Documents\GitHub\Holodeck-Time-Travel-Task-Analytics\
# Visualizer\001_1_1_1_2016-08-29_10-26-03.dat"
parser = argparse.ArgumentParser(description='When called without arguments, a file dialog will appear to select ' +
                                             'a log file. When called with arguments, a log file can be selected via ' +
                                             'the first argument.')
parser.add_argument('--log_file', dest='log_file',
                    help='string path to a log file from the Holodeck Time Travel Task')

args = parser.parse_args()

########################################################################################################################
# Get Log File Path and Load File
########################################################################################################################

local_directory = os.path.dirname(os.path.realpath(__file__))  # The directory of this script
# filename = '001_1_1_1_2016-08-29_10-26-03.dat'  # The relative path to the data file (CHANGE ME)
# path = os.path.join(local_directory, filename)
if args.log_file is None:
    root = tkinter.Tk()
    root.withdraw()
    path = tkFileDialog.askopenfilename()
else:
    path = args.log_file

meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

logging.info("Parsing file (" + str(path) + ")...")
# First we populate a list of each iteration's data
# This section of code contains some custom binary parser data which won't be explained here
iterations = read_binary_file(path)
# Output the iterations count for debugging purposes
logging.info("Plotting " + str(len(iterations)) + " iterations.")

########################################################################################################################
# Generate UI Window and Set Camera Settings
########################################################################################################################

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['center'] = pg.Qt.QtGui.QVector3D(0, 0, 30)
w.opts['distance'] = 200
w.setWindowTitle('Timeline Visualizer')

########################################################################################################################
# Make Grid
########################################################################################################################


grid_items = []


def make_grid_item(loc, rot, scale):
    global grid_items
    g = gl.GLGridItem()
    g.scale(scale[0], scale[1], scale[2])
    g.rotate(rot[0], rot[1], rot[2], rot[3])
    g.translate(loc[0], loc[1], loc[2])
    grid_items.append(g)
    return g


if meta['phase'] == '0':
    w.addItem(make_grid_item((-19, 0, 15), (90, 0, 1, 0), (1.5, 1.9, 1.9)))
    w.addItem(make_grid_item((0, -19, 15), (90, 1, 0, 0), (1.9, 1.5, 1.9)))
else:
    w.addItem(make_grid_item((-19, 0, 15), (90, 0, 1, 0), (1.5, 1.9, 1.9)))
    w.addItem(make_grid_item((-19, 0, 45), (90, 0, 1, 0), (1.5, 1.9, 1.9)))
    w.addItem(make_grid_item((0, -19, 15), (90, 1, 0, 0), (1.9, 1.5, 1.9)))
    w.addItem(make_grid_item((0, -19, 45), (90, 1, 0, 0), (1.9, 1.5, 1.9)))
w.addItem(make_grid_item((0, 0, 0), (0, 0, 0, 0), (1.9, 1.9, 1.9)))

########################################################################################################################
# Make Image Base
########################################################################################################################

# Determine the background image according to meta phase
bg_path = 'studyBG.png'
if meta['phase'] == '0':
    bg_path = 'practiceBG.png'
img = imread(os.path.join(local_directory, bg_path))

image_scale = (19.0 * 2.0) / float(img.shape[0])
tex1 = pg.makeRGBA(img)[0]
base_image = gl.GLImageItem(tex1)
base_image.translate(-19, -19, 0)
base_image.rotate(270, 0, 0, 1)
base_image.scale(image_scale, image_scale, image_scale)
w.addItem(base_image)

########################################################################################################################
# Make Timeline Colored Bars
########################################################################################################################


color_bars = []


def make_color_bar(rgb, p, r, s):
    global color_bars
    col = np.array([[rgb]])
    tex = pg.makeRGBA(col)[0]
    v = gl.GLImageItem(tex)
    v.translate(p[0], p[1], p[2])
    v.scale(s[0], s[1], s[2])
    v.rotate(r[0], r[1], r[2], r[3])
    color_bars.append(v)
    return v


color_bar_length = 15
if meta['phase'] == '0':
    times = [0, 7.5, 15, 22.5]
    color_bar_length = 7.5
else:
    times = [0, 15, 30, 45]
if meta['inverse'] == '1':
    times.reverse()

w.addItem(make_color_bar((255, 255, 0), (19, times[0], 19), (90, 1, 0, 0), (5, color_bar_length, 0)))
w.addItem(make_color_bar((255, 0, 0), (19, times[1], 19), (90, 1, 0, 0), (5, color_bar_length, 0)))
w.addItem(make_color_bar((0, 255, 0), (19, times[2], 19), (90, 1, 0, 0), (5, color_bar_length, 0)))
w.addItem(make_color_bar((0, 0, 255), (19, times[3], 19), (90, 1, 0, 0), (5, color_bar_length, 0)))

########################################################################################################################
# Generate Path Line
########################################################################################################################

forwardColor = (255, 255, 255, 255)
backwardColor = (255, 0, 255, 255)
line_color = np.empty((len(iterations), 4))
line_color_state = np.empty((len(iterations), 4))
x = []
y = []
z = []
for idx, i in enumerate(iterations):
    x.append(float(i['x']))
    y.append(float(i['z']))
    z.append(float(i['time']))
    c = forwardColor
    if i['timescale'] <= 0:
        c = backwardColor
    line_color[idx] = pg.glColor(c)
    line_color_state[idx] = pg.glColor((0, 0, 0, 0))

pts = np.vstack([x, y, z]).transpose()
path_line = gl.GLLinePlotItem(pos=pts, color=line_color_state, mode='line_strip', antialias=True)
w.addItem(path_line)

########################################################################################################################
# Generate Item Lines (ground truth)
########################################################################################################################

if meta['phase'] == '0':
    times = [2, 12, 18, 25]
    directions = [1, 0, 1, 0]  # Fall = 1, Fly = 0, Stay = 2
    if meta['inverse'] == '1':
        times.reverse()
        directions.reverse()
    items = [{'direction': directions[0], 'pos': (2, -12, times[0]), 'color': (255, 255, 0)},
             {'direction': directions[1], 'pos': (2, 13, times[1]), 'color': (255, 0, 0)},
             {'direction': directions[2], 'pos': (-13, 2, times[2]), 'color': (0, 255, 0)},
             {'direction': directions[3], 'pos': (-12, -17, times[3]), 'color': (0, 0, 255)},
             {'direction': 2, 'pos': (13, 5, 0), 'color': (128, 0, 128)}]
else:
    times = [4, 10, 16, 25, 34, 40, 46, 51]
    directions = [1, 0, 0, 1, 1, 0, 1, 0]  # Fall = 1, Fly = 0, Stay = 2
    if meta['inverse'] == '1':
        times.reverse()
        directions.reverse()
    items = [{'direction': directions[0], 'pos': (18, -13, times[0]), 'color': (255, 255, 0)},
             {'direction': directions[1], 'pos': (-13, 9, times[1]), 'color': (255, 255, 0)},
             {'direction': directions[2], 'pos': (-10, -2, times[2]), 'color': (255, 0, 0)},
             {'direction': directions[3], 'pos': (6, -2, times[3]), 'color': (255, 0, 0)},
             {'direction': directions[4], 'pos': (17, -8, times[4]), 'color': (0, 255, 0)},
             {'direction': directions[5], 'pos': (-2, -7, times[5]), 'color': (0, 255, 0)},
             {'direction': directions[6], 'pos': (-15, -15, times[6]), 'color': (0, 0, 255)},
             {'direction': directions[7], 'pos': (6, 18, times[7]), 'color': (0, 0, 255)},
             {'direction': 2, 'pos': (14, 6, 0), 'color': (128, 0, 128)},
             {'direction': 2, 'pos': (-2, 10, 0), 'color': (128, 0, 128)}]

item_lines = []
pos = np.empty((len(items), 3))
size = np.empty((len(items)))
color = np.empty((len(items), 4))
end_time = 60
if meta['phase'] == '0':
    end_time = 30
for idx, i in enumerate(items):
    pos[idx] = i['pos']
    size[idx] = 2
    if i['direction'] == 2:
        size[idx] = 0
    color[idx] = (i['color'][0] / 255, i['color'][1] / 255, i['color'][2] / 255, 1)
    idx += 1
    end = i['pos']
    if i['direction'] == 0:
        end = (end[0], end[1], 0)
    elif i['direction'] > 0:
        end = (end[0], end[1], end_time)
    line = gl.GLLinePlotItem(pos=np.vstack([[i['pos'][0], end[0]],
                                            [i['pos'][1], end[1]],
                                            [i['pos'][2], end[2]]]).transpose(),
                             color=pg.glColor(i['color']), width=3, antialias=True)
    item_lines.append(line)
    w.addItem(line)

item_scatter_plot = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
w.addItem(item_scatter_plot)

########################################################################################################################
# Show UI
########################################################################################################################

w.show()
logging.info("Showing plot. Close plot to exit program.")

########################################################################################################################
# Custom Keyboard Controls
########################################################################################################################

# These variables are modified by the keyboard controls
idx = 0
num_points_to_update = 5
saved_points_to_update = 0
paused = False


def speed_up():
    global num_points_to_update, paused
    if not paused:
        num_points_to_update += 5
        logging.info("Setting speed to " + str(num_points_to_update) + " points per tick.")


def speed_down():
    global num_points_to_update, paused
    if not paused:
        num_points_to_update -= 5
        logging.info("Setting speed to " + str(num_points_to_update) + " points per tick.")


def pause():
    global num_points_to_update, saved_points_to_update, paused
    if not paused:
        logging.info("Paused.")
        saved_points_to_update = num_points_to_update
        num_points_to_update = 0
        paused = True
    else:
        logging.info("Unpaused.")
        num_points_to_update = saved_points_to_update
        saved_points_to_update = -0.5
        paused = False


def reset():
    global idx, line_color_state
    logging.info("Resetting to time zero.")
    idx = 0
    for index in range(0, len(line_color_state) - 1):
        line_color_state[index] = (0, 0, 0, 0)


def go_to_end():
    global idx, line_color_state, line_color
    logging.info("Going to end.")
    idx = len(line_color_state) - 1
    for index in range(0, len(line_color_state) - 1):
        line_color_state[index] = line_color[index]


def close_all():
    global timer, app
    logging.info("User Shutdown Via Button Press")
    timer.stop()
    app.closeAllWindows()


# Visibility Variables
grid_visible = True
base_visible = True
color_bars_visible = True
items_visible = True
path_line_visible = True


def toggle_grid_visible():
    global grid_visible
    if grid_visible:
        for g in grid_items:
            g.hide()
        grid_visible = False
    else:
        for g in grid_items:
            g.show()
        grid_visible = True


def toggle_base_visible():
    global base_visible
    if base_visible:
        base_image.hide()
        base_visible = False
    else:
        base_image.show()
        base_visible = True


def toggle_color_bars_visible():
    global color_bars_visible
    if color_bars_visible:
        for b in color_bars:
            b.hide()
        color_bars_visible = False
    else:
        for b in color_bars:
            b.show()
        color_bars_visible = True


def toggle_items_visible():
    global items_visible
    if items_visible:
        item_scatter_plot.hide()
        for il in item_lines:
            il.hide()
        items_visible = False
    else:
        item_scatter_plot.show()
        for il in item_lines:
            il.show()
        items_visible = True


def toggle_path_line_visible():
    global path_line_visible
    if path_line_visible:
        path_line.hide()
        path_line_visible = False
    else:
        path_line.show()
        path_line_visible = True


sh = QtGui.QShortcut(QtGui.QKeySequence("+"), w, speed_up)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("-"), w, speed_down)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence(" "), w, pause)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("R"), w, reset)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("E"), w, go_to_end)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("Escape"), w, close_all)
sh.setContext(QtCore.Qt.ApplicationShortcut)

sh = QtGui.QShortcut(QtGui.QKeySequence("1"), w, toggle_grid_visible)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("2"), w, toggle_base_visible)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("3"), w, toggle_color_bars_visible)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("4"), w, toggle_items_visible)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("5"), w, toggle_path_line_visible)
sh.setContext(QtCore.Qt.ApplicationShortcut)


########################################################################################################################
# Animation Loop
########################################################################################################################


def update():
    global path_line, idx, timer
    for _ in range(0, abs(num_points_to_update)):
        if num_points_to_update > 0:
            line_color_state[idx] = line_color[idx]
            idx += 1
        else:
            line_color_state[idx] = (0, 0, 0, 0)
            idx -= 1
        if idx < 0:
            idx = 0
        elif idx >= len(line_color):
            idx = len(line_color) - 1
            break
    path_line.setData(color=line_color_state)


timer = QtCore.QTimer()
# noinspection PyUnresolvedReferences
timer.timeout.connect(update)
timer.start(1)

########################################################################################################################
# PyQtGraph Initialization
########################################################################################################################

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # noinspection PyArgumentList
        QtGui.QApplication.instance().exec_()
