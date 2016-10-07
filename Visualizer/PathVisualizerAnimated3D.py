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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

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
if path is '':
    logging.info('No file selected. Closing.')
    exit()
if not os.path.exists(path):
    logging.error('File not found. Closing.')
    exit()

meta = None
# noinspection PyBroadException
try:
    meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
except:
    logging.error('There was an error reading the filename meta-information. Please confirm this is a valid log file.')
    exit()

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
    tmp_texture = pg.makeRGBA(col)[0]
    v = gl.GLImageItem(tmp_texture)
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
    directions = [2, 1, 2, 1]  # Fall = 2, Fly = 1, Stay = 0
    if meta['inverse'] == '1':
        times.reverse()
        directions.reverse()
    items = [{'direction': directions[0], 'pos': (2, -12, times[0]), 'color': (255, 255, 0)},
             {'direction': directions[1], 'pos': (2, 13, times[1]), 'color': (255, 0, 0)},
             {'direction': directions[2], 'pos': (-13, 2, times[2]), 'color': (0, 255, 0)},
             {'direction': directions[3], 'pos': (-12, -17, times[3]), 'color': (0, 0, 255)},
             {'direction': 0, 'pos': (13, 5, 0), 'color': (128, 0, 128)}]
else:
    times = [4, 10, 16, 25, 34, 40, 46, 51]
    directions = [2, 1, 1, 2, 2, 1, 2, 1]  # Fall = 2, Fly = 1, Stay = 0
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
             {'direction': 0, 'pos': (14, 6, 0), 'color': (128, 0, 128)},
             {'direction': 0, 'pos': (-2, 10, 0), 'color': (128, 0, 128)}]

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
    if i['direction'] == 0:
        size[idx] = 0
    color[idx] = (i['color'][0] / 255, i['color'][1] / 255, i['color'][2] / 255, 1)
    idx += 1
    end = i['pos']
    if i['direction'] == 1:
        end = (end[0], end[1], 0)
    elif i['direction'] == 2 or i['direction'] == 0:
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
# If Test, Generate Reconstruction Items
########################################################################################################################
event_state_labels = ['stationary', 'up', 'down']
item_number_label = ['bottle', 'icecubetray', 'clover', 'basketball', 'boot', 'crown', 'bandana', 'hammer',
                     'fireext', 'guitar']
item_label_filename = ['bottle.jpg', 'icecubetray.jpg', 'clover.jpg', 'basketball.jpg',
                       'boot.jpg', 'crown.jpg', 'bandana.jpg', 'hammer.jpg',
                       'fireextinguisher.jpg', 'guitar.jpg']
descrambler = [1, 2, 4, 7, 0, 3, 5, 6, 8, 9]
descrambler_type = [2, 2, 2, 2, 1, 1, 1, 1, 0, 0]


def find_last(lst, sought_elt):
    for r_idx, elt in enumerate(reversed(lst)):
        if elt == sought_elt:
            return len(lst) - 1 - r_idx


pastel_factor = 127
cols = [(255, 255, pastel_factor), (255, 255, pastel_factor),
        (255, pastel_factor, pastel_factor), (255, pastel_factor, pastel_factor),
        (pastel_factor, 255, pastel_factor), (pastel_factor, 255, pastel_factor),
        (pastel_factor, pastel_factor, 255),
        (pastel_factor, pastel_factor, 255),
        (128, pastel_factor / 2, 128), (128, pastel_factor / 2, 128)]

reconstruction_item_scatter_plot = None
reconstruction_item_lines = []
reconstruction_items = [None] * len(item_number_label)
billboard_item_labels = []

if meta['phase'] == '2':
    pos = np.empty((len(items), 3))
    size = np.empty((len(items)))
    color = np.empty((len(items), 4))
    end_time = 60  # End time for convenience

    ####################################################################################################################
    # BEGIN DESCRAMBLER
    ####################################################################################################################

    start_state = iterations[0]  # Store first iteration
    end_state = iterations[len(iterations) - 1]  # Store last iteration
    prev_active = start_state['itemsactive']  # Create activity array

    # Event state tracker variables (this works great)
    event_state = 0
    prev_event_btn_state = False
    prev_drop_button_state = False
    for iterations_idx, i in enumerate(iterations):
        if iterations_idx == len(iterations) - 1:
            break
        # Get the event state button (keys, 4, buttons, 4)
        event_button_state = i['buttons'][4]
        if event_button_state and not prev_event_btn_state:  # On rising edge
            event_state = (event_state + 1) % 3  # Set the event state appropriately
        prev_event_btn_state = event_button_state  # Update prev state for edge detection
        # Get the item drop button state (keys 1, buttons 1)
        drop_button_state = i['buttons'][1]
        if drop_button_state and not prev_drop_button_state:  # On rising edge, notify user of drop
            logging.debug(
                'drop item as ' + event_state_labels[event_state] + str(i['time']) + ','.join(map(str, i['keys'])))

        # Find the value of the index of this item in the descrambler, this is the correct item value (I think)
        # Coordinate comparison
        current_item_coords = []
        for xx, zz in zip(i['itemsx'], i['itemsz']):
            current_item_coords.append((xx, zz))
        next_iteration = iterations[iterations_idx + 1]
        next_item_coords = []
        for xx, zz in zip(next_iteration['itemsx'], next_iteration['itemsz']):
            next_item_coords.append((xx, zz))
        # Check for changed state (at this point, active and button press simultaneously)
        if (current_item_coords != next_item_coords) and (not iterations_idx == 0):
            logging.debug(','.join(map(str, current_item_coords)))
            logging.debug(','.join(map(str, next_item_coords)))
            present_checklist = [False] * len(current_item_coords)
            ir_index = -1
            missing_item_index = -1
            for idxx, (first, second) in enumerate(zip(current_item_coords, next_item_coords)):
                if first in next_item_coords:
                    present_checklist[idxx] = True
                if not first == second and (not ir_index == -1):
                    ir_index = idxx
            for idxx, check in enumerate(present_checklist):
                if not check:
                    missing_item_index = idxx
            # Now we know which item (according to the previous descrambler state) had its location change
            # and which index it WAS in. We also know the event type it was placed as so we can compute its NEW
            # position in the list.
            # missing_item_index is the index of the placed item (according to current descrambler)
            # event_state is the type of event which was placed
            # Edge cases include if an item is placed precisely back where it was and if multiple items are
            # placed in the same place

            # DESCRAMBLER LOGIC
            # If the current event state is 0 (stationary), move the current item to the end of the list
            insertion_index = -1
            val = descrambler[missing_item_index]
            del descrambler[missing_item_index]
            del descrambler_type[missing_item_index]
            if event_state == 0:
                descrambler.append(val)
                descrambler_type.append(event_state)
                insertion_index = len(descrambler) - 1
            # If the current event state is 1 (up/fly), move the current item to the last fly position
            elif event_state == 1 or event_state == 2:
                last = find_last(descrambler_type, event_state)
                descrambler.insert(last + 1, val)
                descrambler_type.insert(last + 1, event_state)
                insertion_index = last + 1

            # Generate projected values (time is the important one, the space ones are replaced at the end
            # according to the descrambler order)
            placed_x = next_item_coords[insertion_index][0]
            placed_z = next_item_coords[insertion_index][1]
            placed_t = i['time']
            # If the event is stationary, the time of placement doesn't matter, ignore it and set to 0
            if event_state == 0:
                placed_t = 0
            # Add the item to the list using the correct IDX to look up the color
            reconstruction_items[val] = {'direction': event_state,
                                         'pos': (placed_x, placed_z, placed_t),
                                         'color': cols[val]}
            # Log debug information
            logging.debug(','.join(map(str, descrambler)))
            logging.debug("{0}, {1}, ({2}, {3}, {4})".format(
                item_number_label[val].ljust(11, ' '), event_state_labels[event_state], placed_x, placed_z, placed_t))
        prev_drop_button_state = drop_button_state

    # Replace all of the position values with the descrambled position values at the final time point.
    # Keep the time point the same as it should've been corrected earlier
    # for idx in range(0, len(reconstruction_items)):
    #    reconstruction_items[idx]['pos'] = (end_state['itemsx'][descrambler[idx]],
    #                                        end_state['itemsz'][descrambler[idx]],
    #                                        reconstruction_items[idx]['pos'][2])
    #    reconstruction_items[idx]['color'] = cols[idx]

    ####################################################################################################################
    # END DESCRAMBLER
    ####################################################################################################################

    # Iterate through the reconstruction items and visualize them
    for idx, i in enumerate(reconstruction_items):
        pos[idx] = i['pos']
        size[idx] = 2
        if i['direction'] == 0:
            size[idx] = 0
        color[idx] = (i['color'][0] / 255, i['color'][1] / 255, i['color'][2] / 255, 1)
        end = pos[idx]
        if i['direction'] == 1:
            end = (end[0], end[1], 0)
        elif i['direction'] == 2 or i['direction'] == 0:
            end = (end[0], end[1], end_time)
        line = gl.GLLinePlotItem(pos=np.vstack([[pos[idx][0], end[0]],
                                                [pos[idx][1], end[1]],
                                                [pos[idx][2], end[2]]]).transpose(),
                                 color=pg.glColor(i['color']), width=3, antialias=True)
        reconstruction_item_lines.append(line)
        w.addItem(line)

        img_path = item_label_filename[idx]
        img = imread(os.path.join(local_directory, img_path))
        expected_size = 2.0
        image_scale = expected_size / float(img.shape[0])
        offset_param = 0.0 - image_scale / 2 - expected_size / 2
        tex = pg.makeRGBA(img)[0]
        label_image = gl.GLImageItem(tex)
        t = pos[idx][2]
        if i['direction'] == 0:
            t = end_time
        label_image.translate(pos[idx][0] + offset_param, pos[idx][1] + offset_param, t)
        label_image.scale(image_scale, image_scale, image_scale)
        w.addItem(label_image)
        billboard_item_labels.append(label_image)
    reconstruction_item_scatter_plot = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
    w.addItem(reconstruction_item_scatter_plot)

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
reconstruction_item_lines_visible = True
billboard_item_labels_visible = True


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


def toggle_reconstruction_item_lines_visible():
    global reconstruction_item_lines_visible
    if reconstruction_item_lines_visible:
        if reconstruction_item_scatter_plot is not None:
            reconstruction_item_scatter_plot.hide()
        for ril in reconstruction_item_lines:
            ril.hide()
        reconstruction_item_lines_visible = False
    else:
        if reconstruction_item_scatter_plot is not None:
            reconstruction_item_scatter_plot.show()
        for ril in reconstruction_item_lines:
            ril.show()
        reconstruction_item_lines_visible = True


def toggle_billboard_item_labels_visible():
    global billboard_item_labels_visible
    if billboard_item_labels_visible:
        for il in billboard_item_labels:
            il.hide()
            billboard_item_labels_visible = False
    else:
        for il in billboard_item_labels:
            il.show()
            billboard_item_labels_visible = True

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
sh = QtGui.QShortcut(QtGui.QKeySequence("6"), w, toggle_reconstruction_item_lines_visible)
sh.setContext(QtCore.Qt.ApplicationShortcut)
sh = QtGui.QShortcut(QtGui.QKeySequence("7"), w, toggle_billboard_item_labels_visible)
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
