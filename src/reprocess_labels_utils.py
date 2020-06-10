import data_loader
import data_loader_util
import data_utils


MOVE_LABELS = (
    "L", "L'",
    "R", "R'",
    "U", "U'",
    "D", "D'",
    "F", "F'",
    "B", "B'",
)

ROTATE_LABELS = (
    "x", "x'", "x2",
    "y", "y'", "y2",
    "z", "z'", "z2",
)



def make_video_spec_dict(video_specs):
    """
    convert video_specs from list to dictionary, key is thee video name w/o extentions
    """
    video_specs_dict = {}
    for v_spec in video_specs:
        video_specs_dict[''.join(v_spec['filename'].split('.')[:-1])] = v_spec

    return video_specs_dict

def get_label_statistics(dict_label_list, show_label_distribution=True, show_duration=True, print_stats=True):
    import collections
    import statistics
    import matplotlib.pyplot as plt
    import numpy as np

    label_duration = collections.defaultdict(list)

    for label_fname in dict_label_list:

        label_pairs = dict_label_list[label_fname]
        begin_frame = max(label_pairs[0][0] - 10, 0)

        for end_frame, label_i in label_pairs:

            label_duration[label_i].append(end_frame - begin_frame)
            begin_frame = end_frame


    if show_label_distribution:
        # draw barchart
        x = label_duration.keys()
        y = [len(label_duration[key]) for key in label_duration]
        y_pos = np.arange(len(x))

        plt.bar(y_pos, y, align='center', alpha=0.5)
        plt.xticks(y_pos, x)
        plt.ylabel('Count')
        plt.title('Distribution of Move Labels')

        plt.show()


    if show_duration:
        # draw duration of moves
        n_bins =  50
        y = []
        for key in label_duration:
            if key in MOVE_LABELS:
                y.extend(label_duration[key])
        plt.hist(y, bins=n_bins)
        plt.xlabel("Number of Frames")
        plt.ylabel("Count")
        plt.title("Duration of Move labels excluding rotation")

        plt.show()


        # draw duration of moves
        n_bins =  50
        y = []
        for key in label_duration:
            if key in ROTATE_LABELS:
                y.extend(label_duration[key])
        plt.hist(y, bins=n_bins)
        plt.xlabel("Number of Frames")
        plt.ylabel("Count")
        plt.title("Duration of Rotation labels excluding moves")

        plt.show()

    label_duration_stats = collections.defaultdict(dict)

    for move_label in label_duration:

        label_duration_stats[move_label]['mean'] = statistics.mean(label_duration[move_label])
        label_duration_stats[move_label]['mode'] = max(set(label_duration[move_label]), key=label_duration[move_label].count)
        label_duration_stats[move_label]['median'] = statistics.median(label_duration[move_label])
        label_duration_stats[move_label]['max'] = max(label_duration[move_label])
        label_duration_stats[move_label]['min'] = min(label_duration[move_label])
        print(move_label,
        " mean:", label_duration_stats[move_label]['mean'],
        "| mode:", label_duration_stats[move_label]['mode'],
        "| median:", label_duration_stats[move_label]['median'],
        "| max:", label_duration_stats[move_label]['max'],
        "| min:", label_duration_stats[move_label]['min'])

        # import matplotlib.pyplot as plt
        # plt.hist(label_duration[move_label], 100)
        # plt.title(move_label)
        # plt.show()

    return label_duration_stats



def convert_label_dict_to_sorted_list(label_dicts):

    dict_label_list = {}

    for label_fname in label_dicts:
        label_list = []

        for index in label_dicts[label_fname]['moves']:
            move_label = label_dicts[label_fname]['moves'][index]
            label_list.append([index, move_label])
        
        label_list = sorted(label_list, key=lambda x: x[0], reverse=False)
        
        dict_label_list[label_fname] = label_list

    return dict_label_list