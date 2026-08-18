import unittest


class MakeCarsCsvTests(unittest.TestCase):
    def test_annotation_rows_are_split_by_official_train_test_flag(self):
        from data.make_cars_csv import annotation_rows

        annotations = [
            ["cars_train/00001.jpg", [[1]], [[2]], [[3]], [[4]], [[7]], [[0]]],
            ["cars_test/00002.jpg", [[10]], [[20]], [[30]], [[40]], [[9]], [[1]]],
        ]

        train_rows, test_rows = annotation_rows(annotations)

        self.assertEqual(train_rows, [[1, 2, 3, 4, 7, "cars_train/00001.jpg"]])
        self.assertEqual(test_rows, [[10, 20, 30, 40, 9, "cars_test/00002.jpg"]])


if __name__ == "__main__":
    unittest.main()
