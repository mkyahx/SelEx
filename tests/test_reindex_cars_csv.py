import unittest


class ReindexCarsCsvTests(unittest.TestCase):
    def test_rows_use_partition_local_five_digit_filenames(self):
        from data.reindex_cars_csv import reindex_rows

        rows = [
            ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class", "relative_im_path"],
            ["112", "7", "853", "717", "1", "car_ims/000001.jpg"],
            ["3", "20", "500", "400", "2", "car_ims/016185.jpg"],
        ]

        converted = reindex_rows(rows)

        self.assertEqual(converted[0], rows[0])
        self.assertEqual(converted[1][-1], "00001.jpg")
        self.assertEqual(converted[2][-1], "00002.jpg")
        self.assertEqual(converted[1][:-1], rows[1][:-1])


if __name__ == "__main__":
    unittest.main()
