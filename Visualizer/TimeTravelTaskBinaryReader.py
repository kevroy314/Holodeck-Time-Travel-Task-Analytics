import datetime
import struct
import logging
import numpy as np
import os
import time
import re
import pytz
import scipy.spatial.distance as distance
from tzlocal import get_localzone
import warnings


# This helper function extracts the meta-data from the filename
def get_filename_meta_data(fn):
    parts = fn.split('_')
    dt = datetime.datetime.strptime(parts[4] + '_' + parts[5].split('.')[0], '%Y-%m-%d_%H-%M-%S')
    return {"subID": parts[0], "trial": parts[1], "phase": parts[2], "inverse": parts[3], "datetime": dt}


# Lookup phase name from number
def phase_num_to_str(phase):
    names = ['VR Practice', 'VR Study', 'VR Test', 'VE Practice', 'VE Study', 'VE Test',
             '2D Practice', '2D Study', '2D Test']
    lookup = phase
    if isinstance(lookup, basestring):
        lookup = int(lookup)
    return names[lookup]


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
    kind = (data % 2 ** 64) >> 62  # This says about UTC and stuff...
    ticks = data & 0x3FFFFFFFFFFFFFFF
    seconds = ticks / 10000000
    tz = pytz.utc
    if kind == 0:
        tz = get_localzone()
    return datetime.datetime(1, 1, 1, tzinfo=tz) + datetime.timedelta(seconds=seconds)


def read_binary_file(path):
    iterations = []
    with open(path, 'rb') as f:
        header_length = decode_7bit_int_length(f)
        header = f.read(header_length)
        split_header = header.split(',')
        if split_header[0] != 'version':  # Beta version with new version prefix
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
                iterations.append({"version": 0,
                                   "datetime": date_time, "time": time, "timescale": time_scale, "x": x, "y": y, "z": z,
                                   "rx": rx, "ry": ry, "rz": rz, "rw": rw,
                                   "keys": keys, "buttons": buttons,
                                   "itemsx": ix, "itemsy": iy, "itemsz": iz, "itemsactive": i_active,
                                   "itemsclicked": i_clicked,
                                   "boundarystate": boundary_state, "br": br, "bg": bg, "bb": bb})
        elif split_header[1] == '2':  # Version 2
            num_keys = header.count('key')
            num_buttons = header.count('button')
            num_items = header.count('itemXYZActiveClickedEventTime')
            key_labels = []
            key_split = header.split('key')
            for i in range(1, len(key_split)):
                key_labels.append(key_split[i].split('_')[0])
            button_labels = []
            button_split = header.split('button')
            for i in range(1, len(button_split)):
                button_labels.append(button_split[i].split('_')[0])
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
                i_event_type = []
                i_event_time = []
                # noinspection PyRedeclaration
                for i in range(0, num_items):
                    ix.append(struct.unpack_from('f', f.read(4))[0])
                    iy.append(struct.unpack_from('f', f.read(4))[0])
                    iz.append(struct.unpack_from('f', f.read(4))[0])
                    i_active.append(struct.unpack_from('?', f.read(1))[0])
                    i_clicked.append(struct.unpack_from('?', f.read(1))[0])
                    i_event_type.append(struct.unpack_from('i', f.read(4))[0])
                    i_event_time.append(struct.unpack_from('f', f.read(4))[0])

                # Extract boundary information
                boundary_state = struct.unpack_from('i', f.read(4))[0]
                br = struct.unpack_from('f', f.read(4))[0]
                bg = struct.unpack_from('f', f.read(4))[0]
                bb = struct.unpack_from('f', f.read(4))[0]

                # Extract inventory state
                inventory_item_numbers = []
                for i in range(0, num_items):
                    inventory_item_numbers.append(struct.unpack_from('i', f.read(4))[0])
                active_inventory_item_number = struct.unpack_from('i', f.read(4))[0]
                active_inventory_event_index = struct.unpack_from('i', f.read(4))[0]

                # Store all information in simple dictionary and add to list of iterations
                iterations.append({"version": 2,
                                   "datetime": date_time, "time": time, "timescale": time_scale, "x": x, "y": y, "z": z,
                                   "rx": rx, "ry": ry, "rz": rz, "rw": rw,
                                   "keys": keys, "buttons": buttons,
                                   'keylabels': key_labels, 'buttonlabels': button_labels,
                                   "itemsx": ix, "itemsy": iy, "itemsz": iz, "itemsactive": i_active,
                                   "itemsclicked": i_clicked, 'itemsevent': i_event_type, 'itemstime': i_event_time,
                                   "boundarystate": boundary_state, "br": br, "bg": bg, "bb": bb,
                                   'inventoryitemnumbers': inventory_item_numbers,
                                   'activeinventoryitemnumber': active_inventory_item_number,
                                   'activeinventoryeventindex': active_inventory_event_index})

        return iterations


def find_last(lst, sought_elt):
    for r_idx, elt in enumerate(reversed(lst)):
        if elt == sought_elt:
            return len(lst) - 1 - r_idx


def parse_test_items(iterations, cols, item_number_label, event_state_labels):
    descrambler = [1, 2, 4, 7, 0, 3, 5, 6, 8, 9]
    descrambler_type = [2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    reconstruction_items = [None] * len(item_number_label)
    billboard_item_labels = []
    if iterations[0]['version'] == 0:
        # pos = np.empty((len(items), 3))
        # size = np.empty((len(items)))
        # color = np.empty((len(items), 4))
        # end_time = 60  # End time for convenience

        ################################################################################################################
        # BEGIN DESCRAMBLER
        ################################################################################################################

        # start_state = iterations[0]  # Store first iteration
        # end_state = iterations[len(iterations) - 1]  # Store last iteration
        # prev_active = start_state['itemsactive']  # Create activity array

        # Event state tracker variables (this works great)
        number_placed = 0
        event_state = 0
        prev_event_btn_state = False
        prev_drop_button_state = False
        prev_inventory_button_state = False
        inventory_index = 0
        inventory = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        numbers_placed = [-1] * 10
        flag_for_same_place_override = False
        flag_for_same_place_location = None
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
            inventory_button_state = i['buttons'][3]
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
                missing_item_index = -1
                for idxx, (first, second) in enumerate(zip(current_item_coords, next_item_coords)):
                    if first in next_item_coords:
                        present_checklist[idxx] = True
                    if next_item_coords.count(first) > 1 or current_item_coords.count(second) > 1:
                        logging.warning('Items were found to be placed on top of each other. ' +
                                        'This will likely make the item identities during reconstruction inaccurate.')
                        logging.debug('CASE: Multiple items in same location, ' +
                                      'consulting Active list for differentiation.')
                        active_list = i['itemsactive']
                        next_active_list = iterations[iterations_idx + 1]['itemsactive']
                        logging.debug(','.join(map(str, active_list)))
                        logging.debug(','.join(map(str, next_active_list)))
                        logging.debug(event_state)
                        logging.debug(','.join(map(str, descrambler_type)))
                        for idxxx, (a, b) in enumerate(zip(current_item_coords, next_item_coords)):
                            if not a == b:
                                present_checklist[idxxx] = False
                                logging.debug('{0} found as move index'.format(idxxx))
                                break
                for idxx, check in enumerate(present_checklist):
                    if not check:
                        if current_item_coords.count(current_item_coords[idxx]) > 1:
                            flag_for_same_place_override = True
                            flag_for_same_place_location = current_item_coords[idxx]
                        missing_item_index = idxx
                logging.debug('{0} is missing item index'.format(missing_item_index))
                if drop_button_state and not prev_drop_button_state:  # Rising edge, item dropped
                    logging.info('item dropped/picked up.')
                    if current_item_coords == next_item_coords:  # item picked up
                        # noinspection PyTypeChecker
                        inventory.insert(inventory_index, '?')
                        logging.info(
                            'item picked up to inventory index {0}: inventory state: {1}'.format(inventory_index,
                                                                                                 inventory))
                    else:
                        if inventory:
                            inventory.pop(inventory_index)
                        numbers_placed[descrambler[missing_item_index]] = number_placed
                        number_placed += 1
                        logging.info(
                            'item dropped from inventory index {0}: inventory state: {1}'.format(inventory_index,
                                                                                                 inventory))
                if inventory_button_state and not prev_inventory_button_state:
                    inventory_index = (inventory_index + 1) % len(inventory)

                # Now we know which item (according to the previous descrambler state) had its location change
                # and which index it WAS in. We also know the event type it was placed as so we can compute its NEW
                # position in the list.
                # missing_item_index is the index of the placed item (according to current descrambler)
                # event_state is the type of event which was placed
                # Edge cases include if an item is placed precisely back where it was and if multiple items are
                # placed in the same place... unfortunately this happens in 44.6% of test files...
                # MORE NOTES:
                # I think it is possible to completely descramble because if an item is inserted into a
                # list of consecutive
                # identical coordinates, the items form a stack (where the latest placed item is picked up first)...
                # when picked up, we now know it's relative index in the inventory (and we can be relatively sure it's
                # picked up because it should become inactive). Since it must be picked up to be placed correctly,
                # if we track its position in the inventory until it is again placed, we can disentangle it from its
                # identical partners. To track it in inventory, it is necessary to track inventory clicks as well as
                # the number of items in the inventory so a proper modulus can be established.
                # Ugh.
                # DESCRAMBLER LOGIC
                if flag_for_same_place_override:
                    tmp_max = -1
                    tmp_max_index = -1
                    for x, y in enumerate(numbers_placed):
                        if current_item_coords[descrambler[x]] == flag_for_same_place_location:
                            if y > tmp_max:
                                tmp_max = y
                                tmp_max_index = x
                    override_index = tmp_max_index
                    # override_index = [x for x, y in enumerate(numbers_placed) if y == max(numbers_placed)][0]
                    override_index_descrambled = [x for x, y in enumerate(descrambler) if y == override_index][0]
                    logging.info('same place override, most recent placed descramble index is {0} compared to '
                                 'original missing index of {1}'.format(override_index_descrambled,
                                                                        missing_item_index))
                    missing_item_index = override_index_descrambled
                    flag_for_same_place_override = False
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
                    if last is None and event_state == 1:
                        last = find_last(descrambler_type, 2)
                    elif last is None and event_state == 2:
                        last = 0
                    logging.debug('inserting into {0}'.format((last + 1)))
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
                    item_number_label[val].ljust(11, ' '),
                    event_state_labels[event_state], placed_x, placed_z, placed_t))
            prev_drop_button_state = drop_button_state
            prev_inventory_button_state = inventory_button_state

            # Replace all of the position values with the descrambled position values at the final time point.
            # Keep the time point the same as it should've been corrected earlier
            # for idx in range(0, len(reconstruction_items)):
            #    reconstruction_items[idx]['pos'] = (end_state['itemsx'][descrambler[idx]],
            #                                        end_state['itemsz'][descrambler[idx]],
            #                                        reconstruction_items[idx]['pos'][2])
            #    reconstruction_items[idx]['color'] = cols[idx]

            ############################################################################################################
            # END DESCRAMBLER
            ############################################################################################################

    if iterations[0]['version'] == 2:
        end_state = iterations[len(iterations) - 1]
        for idx, (x, y, z, active, clicked, event, time) in \
                enumerate(zip(end_state['itemsx'], end_state['itemsy'], end_state['itemsz'], end_state['itemsactive'],
                              end_state['itemsclicked'], end_state['itemsevent'], end_state['itemstime'])):
            reconstruction_items[idx] = {'direction': event, 'pos': (x, z, time), 'color': cols[idx]}

    return billboard_item_labels, reconstruction_items


def get_click_locations_and_indicies(iterations, items, meta):
    # If Study/Practice, label click events
    click_idx = np.empty(len(items))
    click_pos = np.empty((len(items), 3))
    click_size = np.zeros((len(iterations), len(items)))
    click_color = np.empty((len(items), 4))
    if meta['phase'] == '0' or meta['phase'] == '1' or meta['phase'] == '3' or meta['phase'] == '4' \
            or meta['phase'] == '6' or meta['phase'] == '7':
        for idx, i in enumerate(iterations):
            if idx + 1 < len(iterations):
                for idxx, (i1, i2) in enumerate(zip(i['itemsclicked'], iterations[idx + 1]['itemsclicked'])):
                    if i['itemsclicked'][idxx]:
                        click_size[idx][idxx] = 0.5
                    if not i1 == i2:
                        click_idx[idxx] = idx
                        click_pos[idxx] = (i['x'], i['z'], i['time'])
                        click_color[idxx] = (128, 128, 128, 255)
            else:
                for idxx, i1 in enumerate(i['itemsclicked']):
                    if i['itemsclicked'][idxx]:
                        click_size[idx][idxx] = 0.5
    return click_pos, click_idx, click_size, click_color


def get_items_solutions(meta):
    if meta['phase'] == '0' or meta['phase'] == '3' or meta['phase'] == '6':
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
    # elif meta['phase'] == '7' or meta['phase'] == '8':
    #    times = [2, 8, 17, 23]
    #    directions = [2, 1, 1, 2]  # Fall = 2, Fly = 1, Stay = 0
    #    if meta['inverse'] == '1':
    #        times.reverse()
    #        directions.reverse()
    #    items = [{'direction': directions[0], 'pos': (16, -14, times[0]), 'color': (255, 255, 0)},
    #             {'direction': directions[1], 'pos': (-10, -2, times[1]), 'color': (255, 0, 0)},
    #             {'direction': directions[2], 'pos': (15, -8, times[2]), 'color': (0, 255, 0)},
    #             {'direction': directions[3], 'pos': (-15, -15, times[3]), 'color': (0, 0, 255)},
    #             {'direction': 0, 'pos': (-2, 10, 0), 'color': (128, 0, 128)}]
    else:
        times = [4, 10, 16, 25, 34, 40, 46, 51]
        directions = [2, 1, 1, 2, 1, 2, 2, 1]  # Fall = 2, Fly = 1, Stay = 0
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
    return items, times, directions


def find_data_files_in_directory(directory, file_regex=""):
    if not os.path.exists(directory):
        raise IOError('The input path was not found.')

    start_time = time.time()
    data_files = []
    file_index = []
    file_roots_index = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            file_index.append(f)
            file_roots_index.append(root)

    regex = re.compile(file_regex)
    for root, f in zip(file_roots_index, file_index):
            if regex.search(os.path.basename(f)):
                logging.debug('Found data file ({0}).'.format(f))
                data_files.append(os.path.join(root, f))
    logging.info('Found {0} data files in {1} seconds.'.format(len(data_files), time.time() - start_time))
    return data_files


def get_exploration_metrics(iterations):
    total_time = (iterations[-1]['datetime'] - iterations[0]['datetime']).total_seconds()
    space_travelled = 0
    time_travelled = 0
    space_time_travelled = 0
    for idx, i in enumerate(iterations):
        if idx == len(iterations) - 1:
            break
        t = iterations[idx]['time']
        xy = [iterations[idx]['x'], iterations[idx]['y']]
        xyt = xy + [t]
        t_next = iterations[idx + 1]['time']
        xy_next = [iterations[idx + 1]['x'], iterations[idx + 1]['y']]
        xyt_next = xy_next + [t_next]
        space_travelled += distance.euclidean(xy, xy_next)
        space_time_travelled += distance.euclidean(xyt, xyt_next)
        time_travelled += distance.euclidean(t, t_next)

    return total_time, space_travelled, time_travelled, space_time_travelled


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

    space_context_crossing_dist_exclude_wrong_colors_pairs = []
    space_context_noncrossing_dist_exclude_wrong_colors_pairs = []
    space_context_crossing_dist_pairs = []
    space_context_noncrossing_dist_pairs = []

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
        space_dist = distance.euclidean((x0, z0), (x1, z1)) / distance.euclidean((solx0, solz0), (solx1, solz1))
        if crossing:
            context_crossing_dist_pairs.append(dist)
            space_context_crossing_dist_pairs.append(space_dist)
        else:
            context_noncrossing_dist_pairs.append(dist)
            space_context_noncrossing_dist_pairs.append(space_dist)
        if is_correct_color(t0, solt0) and is_correct_color(t1, solt1):
            if crossing:
                context_crossing_dist_exclude_wrong_colors_pairs.append(dist)
                space_context_crossing_dist_exclude_wrong_colors_pairs.append(space_dist)
            else:
                context_noncrossing_dist_exclude_wrong_colors_pairs.append(dist)
                space_context_noncrossing_dist_exclude_wrong_colors_pairs.append(space_dist)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return space_misplacement, time_misplacement, space_time_misplacement, direction_correct_count, \
            np.mean(context_crossing_dist_exclude_wrong_colors_pairs), \
            np.mean(context_noncrossing_dist_exclude_wrong_colors_pairs), \
            np.mean(context_crossing_dist_pairs), np.mean(context_noncrossing_dist_pairs), \
            np.mean(space_context_crossing_dist_exclude_wrong_colors_pairs), \
            np.mean(space_context_noncrossing_dist_exclude_wrong_colors_pairs), \
            np.mean(space_context_crossing_dist_pairs), np.mean(space_context_noncrossing_dist_pairs)


def get_item_details(pastel_factor=127):
    event_state_labels = ['stationary', 'up', 'down']
    item_number_label = ['bottle', 'icecubetray', 'clover', 'basketball', 'boot', 'crown', 'bandana', 'hammer',
                         'fireext', 'guitar']
    item_label_filename = ['bottle.jpg', 'icecubetray.jpg', 'clover.jpg', 'basketball.jpg',
                           'boot.jpg', 'crown.jpg', 'bandana.jpg', 'hammer.jpg',
                           'fireextinguisher.jpg', 'guitar.jpg']

    cols = [(255, 255, pastel_factor), (255, 255, pastel_factor),
            (255, pastel_factor, pastel_factor), (255, pastel_factor, pastel_factor),
            (pastel_factor, 255, pastel_factor), (pastel_factor, 255, pastel_factor),
            (pastel_factor, pastel_factor, 255),
            (pastel_factor, pastel_factor, 255),
            (128, pastel_factor / 2, 128), (128, pastel_factor / 2, 128)]
    return event_state_labels, item_number_label, item_label_filename, cols