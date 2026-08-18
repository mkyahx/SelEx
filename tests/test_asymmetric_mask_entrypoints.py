import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AsymmetricMaskEntrypointTests(unittest.TestCase):
    def test_training_entrypoint_declares_mask_and_foreground_token_options(self):
        path = os.path.join(
            REPO_ROOT,
            "methods",
            "contrastive_training",
            "asymmetric_mask_training.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            contents = handle.read()

        self.assertIn("--mask_root", contents)
        self.assertIn("--max_foreground_tokens", contents)
        self.assertIn("AsymmetricForegroundEncoder", contents)
        self.assertIn("foreground_features", contents)

    def test_dataset_submit_scripts_use_requested_smoothing_and_finetuning(self):
        expected = {
            "submit_asymmetric_cub.sh": ["--dataset_name cub", "--unsupervised_smoothing 1.0"],
            "submit_asymmetric_scars.sh": [
                "--dataset_name scars",
                "--unsupervised_smoothing 1.0",
                "--grad_from_block 9",
            ],
            "submit_asymmetric_aircraft.sh": [
                "--dataset_name aircraft",
                "--unsupervised_smoothing 0.5",
            ],
        }
        for script_name, required_flags in expected.items():
            path = os.path.join(REPO_ROOT, "bash_scripts", script_name)
            with self.subTest(script_name=script_name):
                with open(path, "r", encoding="utf-8") as handle:
                    contents = handle.read()
                for flag in required_flags:
                    self.assertIn(flag, contents)
                self.assertIn("--mask_root", contents)
                self.assertIn("--max_foreground_tokens 128", contents)
                self.assertIn("SAVE_DIR", contents)
                self.assertIn("EXP_NUM", contents)
                self.assertIn("logfile_${EXP_NUM}_seed_${SEED}.out", contents)


if __name__ == "__main__":
    unittest.main()
