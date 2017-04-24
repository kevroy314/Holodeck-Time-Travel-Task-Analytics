from TimeTravelTaskBinaryReader import *
from itertools import chain, izip
import scipy.spatial.distance as distance
import numpy as np
import warnings


def is_correct_color(t, solution_t):
    bins = 15.0
    lower = float(np.floor(float(solution_t) / bins) * bins)
    upper = float(np.ceil(float(solution_t) / bins) * bins)
    return lower < float(t) < upper


def compute_accuracy(meta, items):
    solution_items, times_solution, directions_solution = get_items_solutions(meta)
    xs = [item['pos'][0] for item in items]
    zs = [item['pos'][1] for item in items]
    times = [item['pos'][2] for item in items]
    directions = [item['direction'] for item in items]
    xs_solution = [item['pos'][0] for item in solution_items]
    zs_solution = [item['pos'][1] for item in solution_items]

    space_misplacement = 0
    time_misplacement = 0
    space_time_misplacement = 0
    direction_correct_count = 0
    for x, z, t, d, solx, solz, solt, sold in zip(xs, zs, times, directions, xs_solution, zs_solution, times_solution,
                                                  directions_solution):
        space_misplacement += distance.euclidean((x, z), (solx, solz))
        time_misplacement += np.abs(t - solt)
        space_time_misplacement += distance.euclidean((x, z, t), (solx, solz, solt))
        direction_correct_count += int(d == sold)

    context_crossing_dist_exclude_wrong_colors_pairs = []
    context_noncrossing_dist_exclude_wrong_colors_pairs = []
    context_crossing_dist_pairs = []
    context_noncrossing_dist_pairs = []

    pairs = [(1, 1, 2), (1, 3, 4), (1, 5, 6), (0, 0, 1), (0, 2, 3), (0, 4, 5), (0, 6, 7)]
    for pair in pairs:
        crossing = pair[0] != 0
        idx0 = pair[1]
        idx1 = pair[2]
        x0, z0, t0, d0 = (xs[idx0], zs[idx0], times[idx0], directions[idx0])
        solx0, solz0, solt0, sold0 = (
        xs_solution[idx0], zs_solution[idx0], times_solution[idx0], directions_solution[idx0])
        x1, z1, t1, d1 = (xs[idx1], zs[idx1], times[idx1], directions[idx1])
        solx1, solz1, solt1, sold1 = (
        xs_solution[idx1], zs_solution[idx1], times_solution[idx1], directions_solution[idx1])
        dist = np.abs(t0 - t1) / np.abs(solt0 - solt1)
        if crossing:
            context_crossing_dist_pairs.append(dist)
        else:
            context_noncrossing_dist_pairs.append(dist)
        if is_correct_color(t0, solt0) and is_correct_color(t1, solt1):
            if crossing:
                context_crossing_dist_exclude_wrong_colors_pairs.append(dist)
            else:
                context_noncrossing_dist_exclude_wrong_colors_pairs.append(dist)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return space_misplacement, time_misplacement, space_time_misplacement, direction_correct_count, \
            np.mean(context_crossing_dist_exclude_wrong_colors_pairs), \
            np.mean(context_noncrossing_dist_exclude_wrong_colors_pairs), \
            np.mean(context_crossing_dist_pairs), np.mean(context_noncrossing_dist_pairs)


files = find_data_files_in_directory('C:\Users\Kevin\Desktop\Work\Time Travel Task',
                                     file_regex="\d\d\d_\d_2_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat")

event_state_labels = ['stationary', 'up', 'down']
item_number_label = ['bottle', 'icecubetray', 'clover', 'basketball', 'boot', 'crown', 'bandana', 'hammer',
                     'fireext', 'guitar']
item_label_filename = ['bottle.jpg', 'icecubetray.jpg', 'clover.jpg', 'basketball.jpg',
                       'boot.jpg', 'crown.jpg', 'bandana.jpg', 'hammer.jpg',
                       'fireextinguisher.jpg', 'guitar.jpg']

pastel_factor = 127
cols = [(255, 255, pastel_factor), (255, 255, pastel_factor),
        (255, pastel_factor, pastel_factor), (255, pastel_factor, pastel_factor),
        (pastel_factor, 255, pastel_factor), (pastel_factor, 255, pastel_factor),
        (pastel_factor, pastel_factor, 255),
        (pastel_factor, pastel_factor, 255),
        (128, pastel_factor / 2, 128), (128, pastel_factor / 2, 128)]

out_path = 'out.csv'
last_pilot_id = 20
fp = open(out_path, 'wb')
item_labels = list(chain.from_iterable(izip([x + '_x' for x in item_number_label],
                                            [x + '_z' for x in item_number_label],
                                            [x + '_time' for x in item_number_label],
                                            [x + '_type' for x in item_number_label])))
header = 'subID,trial,inverse,datetime,placed_items,complete?,pilot?,space_misplacement,time_misplacement,space_time_misplacement,event_type_correct_count,context_crossing_dist_exclude_wrong_color_pairs,context_noncrossing_dist_exclude_wrong_color_pairs,context_crossing_dist_pairs,context_noncrossing_dist_pairs,' + ','.join(
    item_labels)
fp.write(header + '\r\n')
for path in files:
    iterations = read_binary_file(path)
    billboard_item_labels, reconstruction_items = parse_test_items(iterations, cols,
                                                                   item_number_label, event_state_labels)
    meta = get_filename_meta_data(os.path.basename(path))
    line_start = meta['subID'] + ',' + meta['trial'] + ',' + meta['inverse'] + ',' + str(meta['datetime'])
    items = [','.join([str(x) for x in item['pos']]) + ',' + str(item['direction']) if item is not None
             else 'nan,nan,nan,nan' for item in reconstruction_items]
    line_start += ',' + str(len(items) - items.count('nan,nan,nan,nan'))
    complete = items.count('nan,nan,nan,nan') == 0
    line_start += ',' + str(complete)
    line_start += ',' + str(int(meta['subID']) <= last_pilot_id)
    if complete:
        space, time, space_time, event, ccexc, cncexc, cc, cnc = compute_accuracy(meta, reconstruction_items)
    else:
        space, time, space_time, event, ccexc, cncexc, cc, cnc = \
            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    line_start += ',' + str(space) + ',' + str(time) + ',' + str(space_time) + ',' + str(event)
    line_start += ',' + str(ccexc) + ',' + str(cncexc) + ',' + str(cc) + ',' + str(cnc)
    line = line_start + ',' + (','.join(items))
    print(line_start)
    fp.write(line + '\r\n')
    fp.flush()

fp.close()
