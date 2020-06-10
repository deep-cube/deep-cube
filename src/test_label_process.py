from label_process import *
import unittest


class TestLabelProcess(unittest.TestCase):

    def test_cleanup_raw(self):
        raw_label = {
            'moves': [
                ['x y x2', 0, ],
                ['R D', 1000, ],
                ['x', 1000, ],
                ['L', 1034, ],
                ['x y2', 2000],
                ['y2', 2100],  # y2 to be removed
                ['x y2', 2600],
                ['y2', 3000],  # y2 to not be removed
            ]
        }
        result = cleanup_raw(raw_label)
        expected = {
            'moves': [
                ['x y x2', 0, ],
                ['R D', 1000, ],
                ['x', 1000, ],
                ['L', 1034, ],
                ['x', 2000],
                ['x y2', 2600],
                ['y2', 3000],  # y2 to not be removed
            ]
        }
        assert json.dumps(result) == json.dumps(expected)

    def test_process_raw(self):
        raw_label = {
            'moves': [
                ['x y x2', 0, ],
                ['R D', 1000, ],
                ['x', 1000, ],
                ['L', 1034, ],  # 1000ms / 30fps = 33.33333 ms per frame
            ]
        }
        result = process_raw(raw_label)
        expected = {
            'moves': [
                [0, 'x y x2'],
                [30, 'R D'],
                [30, 'x'],
                [31, 'L']
            ]
        }
        assert json.dumps(result) == json.dumps(expected)

    def test_process_framed(self):
        framed_label = {
            'moves': [
                [21, 'x'],
                [24, 'y'],
                [25, 'z'],
                [27, "x' y' x"],
                [27, "L R'"],
                [27, "D"],
                [36, 'x2'],
                [68, 'z2'],
                [98, "R U R'"],
            ]
        }
        result = process_framed(framed_label)
        expected = {
            'moves': {
                21: 'x',
                24: 'y',
                25: 'z',
                27: "x'",
                28: "y'",
                29: "x",
                30: 'L',
                31: "R'",
                32: 'D',
                36: 'x2',
                68: 'z2',
                98: 'R',
                99: 'U',
                100: "R'"
            }
        }

        assert json.dumps(result) == json.dumps(expected)


if __name__ == "__main__":
    unittest.main()
