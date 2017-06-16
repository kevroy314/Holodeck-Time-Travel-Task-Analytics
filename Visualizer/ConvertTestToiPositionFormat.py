from TimeTravelTaskBinaryReader import *
import os


files = find_data_files_in_directory('C:\Users\Kevin\Desktop\Work\Time Travel Task',
                                     file_regex="\d\d\d_\d_2_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat")

event_state_labels, item_number_label, item_label_filename, cols = get_item_details()

path = os.path.dirname(os.path.realpath(__file__))
directory = path+'\\iPositionConversion'
if not os.path.exists(directory):
    os.makedirs(directory)
actual_coords_path = '{0}actual_coordinates.txt'
out_path = '{0}position_data_coordinates.txt'


def save_iposition_items_to_file(filename):
    with open(filename, 'ab') as fp:
        poses = []
        for item in reconstruction_items:
            if item is not None and item['pos'] is not None:
                poses += list(item['pos'])
            else:
                poses += [np.nan, np.nan, np.nan, np.nan]
        line = '\t'.join([str(x) for x in poses])
        fp.write(line + '\r\n')
        fp.flush()

for path in files:
    iterations = read_binary_file(path)
    billboard_item_labels, reconstruction_items = parse_test_items(iterations, cols,
                                                                   item_number_label, event_state_labels)
    meta = get_filename_meta_data(os.path.basename(path))
    items, times, directions = get_items_solutions(meta)

    print(meta['subID'] + ',' + meta['trial'])

    save_iposition_items_to_file(directory+'\\'+out_path.format(meta['subID']))
    save_iposition_items_to_file(directory+'\\'+actual_coords_path.format(meta['subID']))
