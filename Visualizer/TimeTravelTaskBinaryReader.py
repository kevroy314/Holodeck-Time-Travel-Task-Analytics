import datetime
import struct

import pytz
from tzlocal import get_localzone


# This helper function extracts the meta-data from the filename
def get_filename_meta_data(fn):
    parts = fn.split('_')
    dt = datetime.datetime.strptime(parts[4] + '_' + parts[5].split('.')[0], '%Y-%m-%d_%H-%M-%S')
    return {"subID": parts[0], "trial": parts[1], "phase": parts[2], "inverse": parts[3], "datetime": dt}


# From http://stackoverflow.com/questions/1550560/encoding-an-integer-in-7-bit-format-of-c-sharp-binaryreader-readstring
# This function is used in reading the binary files to read the length of the header from the beginning of the file
def decode_7bit_int_length(fp):
    string_length = 0
    string_length_parsed = False
    step = 0
    while not string_length_parsed:
        part = ord(fp.read(1))
        string_length_parsed = ((part >> 7) == 0)
        part_cutter = part & 127
        to_add = part_cutter << (step * 7)
        string_length += to_add
        step += 1
    return string_length


# From http://stackoverflow.com/questions/15919598/serialize-datetime-as-binary
# This function is used in reading the binary files to parse the binary .NET DateTime into a Python datetime
def datetime_from_dot_net_binary(data):
    kind = (data % 2**64) >> 62  # This says about UTC and stuff...
    ticks = data & 0x3FFFFFFFFFFFFFFF
    seconds = ticks/10000000
    tz = pytz.utc
    if kind == 0:
        tz = get_localzone()
    return datetime.datetime(1, 1, 1, tzinfo=tz) + datetime.timedelta(seconds=seconds)


def read_binary_file(path):
    iterations = []
    with open(path, 'rb') as f:
        header_length = decode_7bit_int_length(f)
        header = f.read(header_length)

        num_keys = header.count('key')
        num_buttons = header.count('button')
        num_items = header.count('itemXYZAC')

        while f.read(1):  # Look ahead for end of file
            f.seek(-1, 1)  # Go back one to undo the look-ahead

            # Extract time information
            date_time = datetime_from_dot_net_binary(struct.unpack_from('q', f.read(8))[0])
            time = struct.unpack_from('f', f.read(4))[0]
            time_scale = struct.unpack_from('f', f.read(4))[0]

            # Extract position information
            x = struct.unpack_from('f', f.read(4))[0]
            y = struct.unpack_from('f', f.read(4))[0]
            z = struct.unpack_from('f', f.read(4))[0]

            # Extract rotation information
            rx = struct.unpack_from('f', f.read(4))[0]
            ry = struct.unpack_from('f', f.read(4))[0]
            rz = struct.unpack_from('f', f.read(4))[0]
            rw = struct.unpack_from('f', f.read(4))[0]

            # Extract key, button, and item information according to expected numbers of each
            keys = []
            # noinspection PyRedeclaration
            for i in range(0, num_keys):
                keys.append(struct.unpack_from('?', f.read(1))[0])
            buttons = []
            # noinspection PyRedeclaration
            for i in range(0, num_buttons):
                buttons.append(struct.unpack_from('?', f.read(1))[0])
            ix = []
            iy = []
            iz = []
            i_active = []
            i_clicked = []
            # noinspection PyRedeclaration
            for i in range(0, num_items):
                ix.append(struct.unpack_from('f', f.read(4))[0])
                iy.append(struct.unpack_from('f', f.read(4))[0])
                iz.append(struct.unpack_from('f', f.read(4))[0])
                i_active.append(struct.unpack_from('?', f.read(1))[0])
                i_clicked.append(struct.unpack_from('?', f.read(1))[0])

            # Extract boundary information
            boundary_state = struct.unpack_from('i', f.read(4))[0]
            br = struct.unpack_from('f', f.read(4))[0]
            bg = struct.unpack_from('f', f.read(4))[0]
            bb = struct.unpack_from('f', f.read(4))[0]

            # Store all information in simple dictionary and add to list of iterations
            iterations.append({"datetime": date_time, "time": time, "timescale": time_scale, "x": x, "y": y, "z": z,
                               "rx": rx, "ry": ry, "rz": rz, "rw": rw,
                               "keys": keys, "buttons": buttons,
                               "itemsx": ix, "itemsy": iy, "itemsz": iz, "itemsactive": i_active,
                               "itemsclicked": i_clicked,
                               "boundarystate": boundary_state, "br": br, "bg": bg, "bb": bb})
        return iterations
