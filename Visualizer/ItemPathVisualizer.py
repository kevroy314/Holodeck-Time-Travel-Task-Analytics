import argparse
import logging
import os
import tkFileDialog
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
import matplotlib as mpl
import mpl_toolkits.mplot3d
import numpy as np
import matplotlib.pyplot as plt
from TimeTravelTaskBinaryReader import *
import sklearn.preprocessing as skpre
from scipy import stats
from mayavi import mlab


########################################################################################################################
# Setup
########################################################################################################################

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

normalize_translation = True
normalize_length = True
normalize_rotation = True


def get_rotation_matrix(i_v, unit=None):
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    i_v = np.divide(i_v, np.sqrt(np.dot(i_v, i_v)))
    u, v, w = np.cross(i_v, unit)
    axis = np.array([u, v, w])
    u, v, w = np.divide(axis, np.sqrt(np.dot(axis, axis)))
    d = np.dot(i_v, unit)
    phi = np.arccos(d)
    rcos = np.cos(phi)
    rsin = np.sin(phi)
    matrix = np.zeros((3, 3))
    matrix[0][0] = rcos + u * u * (1.0 - rcos)
    matrix[1][0] = w * rsin + v * u * (1.0 - rcos)
    matrix[2][0] = -v * rsin + w * u * (1.0 - rcos)
    matrix[0][1] = -w * rsin + u * v * (1.0 - rcos)
    matrix[1][1] = rcos + v * v * (1.0 - rcos)
    matrix[2][1] = u * rsin + w * v * (1.0 - rcos)
    matrix[0][2] = v * rsin + u * w * (1.0 - rcos)
    matrix[1][2] = -u * rsin + v * w * (1.0 - rcos)
    matrix[2][2] = rcos + w * w * (1.0 - rcos)
    return matrix


def generate_normed_segments(path, meta=None):
    iterations = read_binary_file(path)
    logging.info("Plotting " + str(len(iterations)) + " iterations.")
    if meta is None:
        # noinspection PyBroadException
        try:
            meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
        except:
            logging.error(
                'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
            exit()
    items, times, directions = get_items_solutions(meta)
    # noinspection PyRedeclaration
    click_pos, click_idx, _, _ = get_click_locations_and_indicies(iterations, items, meta)
    click_idx, click_pos = [list(l) for l in zip(*sorted(zip(click_idx, click_pos)))]
    num_lines = len(click_idx) - 1
    xs = []
    ys = []
    zs = []
    for line_idx in range(0, num_lines):
        start_idx = int(click_idx[line_idx])
        end_idx = int(click_idx[line_idx + 1])
        start_iter = iterations[start_idx]
        end_iter = iterations[end_idx - 1]
        start_pos = [float(start_iter['x']), float(start_iter['z']), float(start_iter['time'])]
        end_pos = [float(end_iter['x']), float(end_iter['z']), float(end_iter['time'])]

        original_vector = np.subtract(end_pos, start_pos)
        magnitude = np.sqrt(np.dot(original_vector, original_vector))

        R = get_rotation_matrix(original_vector)

        x = []
        y = []
        z = []
        sub_iterations = iterations[start_idx:end_idx]
        for idx, i in enumerate(sub_iterations):
            xtmp = float(i['x'])
            ytmp = float(i['z'])
            ztmp = float(i['time'])
            if normalize_translation:
                xtmp, ytmp, ztmp = np.subtract([xtmp, ytmp, ztmp], start_pos)
            if normalize_length:
                xtmp, ytmp, ztmp = np.divide([xtmp, ytmp, ztmp], magnitude)
            if normalize_rotation:
                xtmp, ytmp, ztmp = np.dot(np.array([xtmp, ytmp, ztmp]).T, R.T)
            x.append(xtmp)
            y.append(ytmp)
            z.append(ztmp)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs, num_lines

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

# Have a look at the colormaps here and decide which one you'd like:
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.Accent  # winter

files = find_data_files_in_directory('C:\Users\Kevin\Desktop\Work\Time Travel Task', file_regex="\d\d\d_\d_1_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat")


def subsetter(path):
    meta = None
    # noinspection PyBroadException
    try:
        meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
    except:
        logging.error(
            'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
        exit()
    return int(meta["subID"]) == 29 and int(meta["trial"]) >= 0

num_subs = 0
for path in files:
    if subsetter(path):
        num_subs += 1

count = 0
for path in files:
    meta = None
    # noinspection PyBroadException
    try:
        meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
    except:
        logging.error(
            'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
        exit()
    if subsetter(path):
        xs, ys, zs, num_lines = generate_normed_segments(path, meta)
        # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_lines)])
        base_colors = [colormap(i) for i in np.linspace(0, 0.9, num_subs)]
        col_space = [base_colors[count]]*num_lines
        col_space = [(x[0], x[1], x[2], a) for x, a in zip(col_space, np.linspace(0.5, 1, num_lines))]
        labels = [""]*num_lines
        labels[-1] = "{0} trial {1}".format(meta["subID"], meta["trial"])
        [ax.plot(x, y, z, label=labels[idx], color=col_space[idx]) for idx, (x, y, z) in enumerate(zip(xs, ys, zs))]
        count += 1

ax.scatter([0, 1], [0, 0], [0, 0], color='k')
ax.legend()
ax.set_aspect('equal')
plt.show()

'''xss = []
yss = []
zss = []
for path in files:
    meta = None
    # noinspection PyBroadException
    try:
        meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
    except:
        logging.error(
            'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
        exit()
    if subsetter(path):
        xs, ys, zs, num_lines = generate_normed_segments(path, meta)
        xss.extend(xs)
        yss.extend(ys)
        zss.extend(zs)

xss = np.array([item for sublist in xss for item in sublist])
yss = np.array([item for sublist in yss for item in sublist])
zss = np.array([item for sublist in zss for item in sublist])

xyz = np.vstack([xss, yss, zss])
kde = stats.gaussian_kde(xyz)
# Evaluate kde on a grid
xmin, ymin, zmin = xss.min(), yss.min(), zss.min()
xmax, ymax, zmax = xss.max(), yss.max(), zss.max()
xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
density = kde(coords).reshape(xi.shape)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')

grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
min = density.min()
max=density.max()
mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

mlab.axes()
mlab.show()
'''