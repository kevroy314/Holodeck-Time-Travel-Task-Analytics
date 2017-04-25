from TimeTravelTaskBinaryReader import *
from itertools import chain, izip
import numpy as np

files = find_data_files_in_directory('C:\Users\Kevin\Desktop\Work\Time Travel Task',
                                     file_regex="\d\d\d_\d_1_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat")

out_path = 'out_study.csv'
last_pilot_id = 20
fp = open(out_path, 'wb')
header = 'subID,trial,inverse,datetime,pilot?,total_time,space_travelled,time_travelled,space_time_travelled'
fp.write(header + '\r\n')
for path in files:
    iterations = read_binary_file(path)
    total_time, space_travelled, time_travelled, space_time_travelled = get_exploration_metrics(iterations)
    meta = get_filename_meta_data(os.path.basename(path))
    line_start = meta['subID'] + ',' + meta['trial'] + ',' + meta['inverse'] + ',' + str(meta['datetime'])
    line_start += ',' + str(int(meta['subID']) <= last_pilot_id)
    line = line_start + ',' + str(total_time) + ',' + str(space_travelled) + ',' \
           + str(time_travelled) + ',' + str(space_time_travelled)
    print(line_start)
    fp.write(line + '\r\n')
    fp.flush()

fp.close()
