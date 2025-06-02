import unittest
import vmbpy

from slmsuite.hardware.cameras.alliedvision import AlliedVision

max_woi = (0, 2464, 0, 2064)

class TestAlliedVision(unittest.TestCase):

    def setUp(self):
        self.cam = AlliedVision('08V6K')

    def test_set_woi_width_not_multiple(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([0, 100, 0, 100])
        self.assertIn("Called 'set()' of Feature 'Width' with invalid value. 100 is not a multiple of 8, starting at 8", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_width_above_range(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([0, 2472, 0, 100])
        self.assertIn("Called 'set()' of Feature 'Width' with invalid value. 2472 is not within [8, 2464]", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_height_below_range(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([0, 8, 0, 0])
        self.assertIn("Called 'set()' of Feature 'Height' with invalid value. 0 is not within [8, 2064].", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_height_above_range(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([0, 8, 0, 2065])
        self.assertIn("Called 'set()' of Feature 'Height' with invalid value. 2065 is not within [8, 2064].", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_offsetx_not_multiple(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([17, 8, 0, 8])
        self.assertIn("Called 'set()' of Feature 'OffsetX' with invalid value. 17 is not a multiple of 16, starting at 0", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_offsetx_above_range(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([2472, 8, 0, 8])
        self.assertIn("Called 'set()' of Feature 'OffsetX' with invalid value. 2472 is not within [0, 2448].", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_offsety_not_multiple(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([0, 8, 1, 8])
        self.assertIn("Called 'set()' of Feature 'OffsetY' with invalid value. 1 is not a multiple of 8, starting at 0", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi_offsety_above_range(self):
        with self.assertRaises(vmbpy.error.VmbFeatureError) as ctx:
            self.cam.set_woi([0, 8, 2064, 8])
        self.assertIn("Called 'set()' of Feature 'OffsetY' with invalid value. 2064 is not within [0, 2056]", str(ctx.exception))
        self.assertEqual(max_woi, self.cam.woi)

    def test_set_woi(self):
        cases = [
            None,
            [0, 8, 0, 8],
            [0, 8, 0, 100],
            [0, 8 * 308, 0, 100],
            [0, 8, 0, 2064],
            [16, 8, 0, 8],
            [2465, 2, 0, 8],
        ]
        for woi in cases:
            with self.subTest(woi=woi):
                self.cam.set_woi(woi)
                self.assertEqual(woi, self.cam.woi)


    def tearDown(self):
        self.cam.close()

if __name__ == '__main__':
    unittest.main()