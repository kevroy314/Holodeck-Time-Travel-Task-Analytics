from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import os
import logging
from scipy.misc import imread
from TimeTravelTaskBinaryReader import read_binary_file, get_filename_meta_data

local_directory = os.path.dirname(os.path.realpath(__file__))  # The directory of this script
filename = '001_1_1_1_2016-08-29_10-26-03.dat'  # The relative path to the data file (CHANGE ME)
meta = get_filename_meta_data(filename)  # The meta filename information for convenience
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

# First we populate a list of each iteration's data
# This section of code contains some custom binary parser data which won't be explained here
iterations = read_binary_file(local_directory, filename)
# Output the iterations count for debugging purposes
logging.info("Plotting " + str(len(iterations)) + " iterations.")

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['center'] = pg.Qt.QtGui.QVector3D(0, 0, 30)
w.opts['distance'] = 200
w.setWindowTitle('Timeline Visualizer')

gx = gl.GLGridItem()
gx.scale(3, 1.9, 1.9)
gx.rotate(90, 0, 1, 0)
gx.translate(-19, 0, 30)
w.addItem(gx)
gy = gl.GLGridItem()
gy.scale(1.9, 3, 1.9)
gy.rotate(90, 1, 0, 0)
gy.translate(0, -19, 30)
w.addItem(gy)
gz = gl.GLGridItem()
gz.scale(1.9, 1.9, 1.9)
gz.translate(0, 0, 0)
w.addItem(gz)

# Determine the background image according to meta phase
bg_path = 'studyBG.png'
if meta['phase'] == '0':
    bg_path = 'practiceBG.png'
img = imread(os.path.join(local_directory, bg_path))

image_scale = (19.0*2.0)/float(img.shape[0])
tex1 = pg.makeRGBA(img)[0]
v1 = gl.GLImageItem(tex1)
v1.translate(-19, -19, 0)
v1.rotate(270, 0, 0, 1)
v1.scale(image_scale, image_scale, image_scale)
w.addItem(v1)


def make_color_bar(rgb, p, r, s):
    col = np.array([[rgb]])
    tex = pg.makeRGBA(col)[0]
    v = gl.GLImageItem(tex)
    v.translate(p[0], p[1], p[2])
    v.scale(s[0], s[1], s[2])
    v.rotate(r[0], r[1], r[2], r[3])
    return v

times = [0, 15, 30, 45]
if meta['inverse'] == '1':
    times.reverse()

w.addItem(make_color_bar((255, 255, 0), (19, times[0], 19), (90, 1, 0, 0), (5, 15, 0)))
w.addItem(make_color_bar((255, 0, 0), (19, times[1], 19), (90, 1, 0, 0), (5, 15, 0)))
w.addItem(make_color_bar((0, 255, 0), (19, times[2], 19), (90, 1, 0, 0), (5, 15, 0)))
w.addItem(make_color_bar((0, 0, 255), (19, times[3], 19), (90, 1, 0, 0), (5, 15, 0)))

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
plt = gl.GLLinePlotItem(pos=pts, color=line_color, mode='line_strip', antialias=True)
w.addItem(plt)

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
         {'direction': 2, 'pos': (-2, 10, 0), 'color': (128, 0, 128)}, ]
pos = np.empty((len(items), 3))
size = np.empty((len(items)))
color = np.empty((len(items), 4))
for idx, i in enumerate(items):
    pos[idx] = i['pos']
    size[idx] = 2
    if i['direction'] == 2:
        size[idx] = 0
    color[idx] = (i['color'][0]/255, i['color'][1]/255, i['color'][2]/255, 1)
    idx += 1
    end = i['pos']
    if i['direction'] == 0:
        end = (end[0], end[1], 0)
    elif i['direction'] > 0:
        end = (end[0], end[1], 60)
    w.addItem(gl.GLLinePlotItem(pos=np.vstack([[i['pos'][0], end[0]],
                                              [i['pos'][1], end[1]],
                                              [i['pos'][2], end[2]]]).transpose(),
                                color=pg.glColor(i['color']), width=3, antialias=True))

sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
w.addItem(sp1)

w.show()
logging.info("Showing plot. Close plot to exit program.")

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
    for index in range(0, len(line_color_state)-1):
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


def update():
    global plt, idx, timer
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
    plt.setData(color=line_color_state)
timer = QtCore.QTimer()
# noinspection PyUnresolvedReferences
timer.timeout.connect(update)
timer.start(1)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # noinspection PyArgumentList
        QtGui.QApplication.instance().exec_()
