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

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['center'] = pg.Qt.QtGui.QVector3D(0, 0, 30)
w.opts['distance'] = 200
w.show()
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
    c = np.array([[rgb]])
    tex = pg.makeRGBA(c)[0]
    v = gl.GLImageItem(tex)
    v.translate(p[0], p[1], p[2])
    v.scale(s[0], s[1], s[2])
    v.rotate(r[0], r[1], r[2], r[3])
    return v

w.addItem(make_color_bar((255, 255, 0), (19, 0, 19), (90, 1, 0, 0), (5, 15, 0)))
w.addItem(make_color_bar((255, 0, 0), (19, 15, 19), (90, 1, 0, 0), (5, 15, 0)))
w.addItem(make_color_bar((0, 255, 0), (19, 30, 19), (90, 1, 0, 0), (5, 15, 0)))
w.addItem(make_color_bar((0, 0, 255), (19, 45, 19), (90, 1, 0, 0), (5, 15, 0)))

# First we populate a list of each iteration's data
# This section of code contains some custom binary parser data which won't be explained here
iterations = read_binary_file(local_directory, filename)

white = pg.mkColor("FFFFFF")

x = []
y = []
z = []
for i in iterations:
    x.append(float(i['x']))
    y.append(float(i['z']))
    z.append(float(i['time']))

pts = np.vstack([x, y, z]).transpose()
plt = gl.GLLinePlotItem(pos=pts, antialias=True)
w.addItem(plt)

# Fall = 1, Fly = 0, Stay = 2
items = [{'direction': 1, 'pos': (18, -13, 4), 'color': (255, 255, 0)},
         {'direction': 0, 'pos': (-13, 9, 10), 'color': (255, 255, 0)},
         {'direction': 0, 'pos': (-10, -2, 16), 'color': (255, 0, 0)},
         {'direction': 1, 'pos': (6, -2, 25), 'color': (255, 0, 0)},
         {'direction': 1, 'pos': (17, -8, 34), 'color': (0, 255, 0)},
         {'direction': 0, 'pos': (-2, -7, 40), 'color': (0, 255, 0)},
         {'direction': 1, 'pos': (-15, -15, 46), 'color': (0, 0, 255)},
         {'direction': 0, 'pos': (6, 18, 51), 'color': (0, 0, 255)},
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


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # noinspection PyArgumentList
        QtGui.QApplication.instance().exec_()
