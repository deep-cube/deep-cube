import numpy as np

MOVE_STRS = [
    "x", "x'", "x2",
    "y", "y'", "y2",
    "z", "z'", "z2",
    "L", "L'",
    "R", "R'",
    "U", "U'",
    "D", "D'",
    "F", "F'",
    "B", "B'",
]

NO_MOVE_STR = "_"

CLASS_STRS = [NO_MOVE_STR] + MOVE_STRS   # _ denotes no movement
NUM_CLASS = len(CLASS_STRS)

CLASS_STR_TO_IDX = {}
IDX_TO_CLASS_STR = {}
for i, s in enumerate(CLASS_STRS):
    CLASS_STR_TO_IDX[s] = i
    IDX_TO_CLASS_STR[i] = s

assert CLASS_STR_TO_IDX[NO_MOVE_STR] == 0  # sanity


def sparse_to_expanded_array(move_dict, start_frame, end_frame, clip_length):
    y = np.zeros(
        shape=(clip_length,),
        dtype=np.uint8
    )
    for i in move_dict:
        if i >= start_frame and i < end_frame:
            y[i - start_frame] = CLASS_STR_TO_IDX[move_dict[i]]
    return y, clip_length


def sparse_to_collapsed_array(move_dict, start_frame, end_frame, clip_length):
    y = []

    last_frame = 0
    for i in move_dict:
        # assume move_dict is sorted
        if i >= start_frame and i < end_frame:
            assert i > last_frame
            last_frame = i
            y.append(CLASS_STR_TO_IDX[move_dict[i]])

    y = np.array(y, dtype=np.uint8)
    target_len = len(y)
    y = np.pad(y, ((0, clip_length - target_len)), constant_values=NUM_CLASS)
    return y, target_len


def class_loss_weights(label_counts):
    # if there is no occurrence of a move, give it 1
    weights = np.ones(NUM_CLASS)

    for label_str in label_counts:
        weights[CLASS_STR_TO_IDX[label_str]] = label_counts[label_str]

    weights = 1 / weights
    weights = weights / weights[0]
    weights = weights ** 0.25
    return weights


def remove_space(y):
    return list(filter(lambda a: a != 0, y))


def ctc_collapse_to_list(y):
    prev = CLASS_STR_TO_IDX[NO_MOVE_STR]
    l = []
    for i in range(len(y)):
        curr = y[i]
        if curr == prev or curr == CLASS_STR_TO_IDX[NO_MOVE_STR]:
            pass
        else:
            l.append(curr)
        prev = curr
    return l


def to_string_list(y):
    return [
        IDX_TO_CLASS_STR[a] for a in y
        if a in IDX_TO_CLASS_STR
    ]


def to_collapsed_string_list(y):
    return list(filter(lambda a: a != NO_MOVE_STR, [
        IDX_TO_CLASS_STR[a] for a in y
        if a in IDX_TO_CLASS_STR
    ]))


def edit_distance(l1, l2):
    if len(l1) > len(l2):
        l1, l2 = l2, l1

    distances = range(len(l1) + 1)
    for i2, c2 in enumerate(l2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(l1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


if __name__ == "__main__":
    from pprint import pprint
    # pprint(CLASS_STR_TO_IDX)
    # pprint(IDX_TO_CLASS_STR)
    # print(NUM_CLASS)

    examples = [
        [],
        [0],
        [1],
        [1, 1, 2],
        [1, 2, 3],
        [0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 2, 1, 1, 0],
        [0, 1, 2, 0, 1, 1, 0],
        [1, 1],
        [1, 0, 1],
    ]
    for example in examples:
        print(np.array(example), ctc_collapse_to_list(example))

    # edit_distance_examples = [
    #     [[1], [1, 2]],
    #     [[1], [1, 1]],
    #     [[1, 1], [1, 2]],
    #     [[], [1]],
    #     [[1, 2, 3], [1, 1, 3]],
    #     [[1, 2, 3], [1, 3, 2]],
    #     [[], []],
    #     [[1], [1]],
    # ]
    # for example in edit_distance_examples:
    #     print(example, edit_distance(*example))
