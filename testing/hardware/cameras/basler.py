import unittest
from pypylon import pylon

from slmsuite.hardware.cameras.basler import Basler

max_woi = (0, 1936, 0, 1216)

class TestBasler(unittest.TestCase):

    def setUp(self):
        self.cam = Basler(serial="24829031")

    def test_set_woi_width_above_range(self):
        with self.assertRaises(pylon.OutOfRangeException) as ctx:
            self.cam.set_woi([0, 2000, 0, 100])
        self.assertIn("Value = 2000 must be equal or smaller than Max = 1936. : OutOfRangeException thrown in node 'Width' while calling 'Width.SetValue()' (file 'IntegerT.h', line 79)", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_height_below_range(self):
        with self.assertRaises(pylon.OutOfRangeException) as ctx:
            self.cam.set_woi([0, 1000, 0, 2000])
        self.assertIn("Value = 2000 must be equal or smaller than Max = 1216. : OutOfRangeException thrown in node 'Height' while calling 'Height.SetValue()' (file 'IntegerT.h', line 79)", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_flush(self):
        self.cam.flush()
        self.assertEqual(self.cam.is_grabbing(), False)

    def tearDown(self):
        self.cam.close()

if __name__ == '__main__':
    unittest.main()