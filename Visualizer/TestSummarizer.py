from TimeTravelTaskBinaryReader import *
from itertools import chain, izip
import scipy.spatial.distance as distance
import numpy as np
import warnings


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

out_path = 'out_test.csv'
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
