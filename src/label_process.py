import os
import sys
import json
from pprint import pprint


FRAMERATE = 30

# if consecutive x2 / y2 / z2 within this time range, remove them
CONSECUTIVE_PACKET_TIME_THRESHOLD_MS = 300
CONSECUTIVE_MOVES_TO_IGNORE = ['x2', 'y2', 'z2']


def cleanup_raw(raw_labels):
    # due to possible sensor noise, sometimes the cube fires consecutive
    # x2 / y2 / z2, and if the packets contain other moves
    # and are received asynchronously out-of-order
    # here we remove consecutive x2 / y2 / z2 s if they're within a time threshold
    # clean this up in-place

    moves_l = [
        (m[0].split(' '), m[1])
        for m in raw_labels['moves']
    ]
    for i in range(len(moves_l)-1):
        lmoves, ltime = moves_l[i]
        rmoves, rtime = moves_l[i+1]
        if rtime - ltime <= CONSECUTIVE_PACKET_TIME_THRESHOLD_MS:
            for m in CONSECUTIVE_MOVES_TO_IGNORE:
                if m in lmoves and m in rmoves:
                    lmoves.remove(m)
                    rmoves.remove(m)

    moves = [
        (' '.join(m[0]), m[1])
        for m in moves_l
        if len(m[0]) > 0
    ]
    return {'moves': moves}


def process_raw(raw_labels):
    # raw_labels: {
    #     moves: [ string, time_in_ms ][]
    # }
    moves = []

    tic = raw_labels['moves'][0][1]  # ms
    for move_str, toc in raw_labels['moves']:
        packet_frame_idx = int((toc - tic) / 1000 * FRAMERATE)
        moves.append((packet_frame_idx, move_str))

    return {'moves': moves}


def process_framed(framed_labels):
    # framed_labels: {
    #     moves: [ number, string ][]
    # }
    # x x are written as x2 for cube rotations
    # R R are written as R R for moves
    moves = {}

    for packet_frame_idx, move_str in framed_labels['moves']:
        # packet_frame_idx assumed to be ordered asc

        while packet_frame_idx in moves:
            # some packets can collide either
            # 1. more than one packet at the same frame
            # 2. previous packet has so many moves that it overflowed
            # shift the colliding current packet to the right until there
            # is space, then lay out moves frame-by-frame
            packet_frame_idx += 1

        for frame_offset, move in enumerate(move_str.split(' ')):
            this_move_frame_idx = packet_frame_idx + frame_offset
            moves[this_move_frame_idx] = move

    return {'moves': moves}


if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    files = [
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f))
    ]
    jsonfiles = [f for f in files if f.split('.')[-1] == 'json']

    for jsonfile in jsonfiles:
        print(f'processing {jsonfile}')
        with open(os.path.join(src_dir, jsonfile)) as f:
            raw_label = json.load(f)
        cleaned_raw_label = cleanup_raw(raw_label)
        framed_label = process_raw(cleaned_raw_label)
        processed_label = process_framed(framed_label)
        with open(os.path.join(dst_dir, jsonfile), 'w') as f:
            json.dump(processed_label, f)
