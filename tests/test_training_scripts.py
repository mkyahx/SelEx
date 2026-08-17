import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TrainingScriptSeedTest(unittest.TestCase):

    def test_contrastive_training_shell_scripts_pass_seed_and_determinism(self):
        script_paths = [
            os.path.join(REPO_ROOT, "contrastive_train.sh"),
            os.path.join(REPO_ROOT, "bash_scripts", "contrastive_train.sh"),
        ]

        for script_path in script_paths:
            with self.subTest(script_path=script_path):
                with open(script_path, "r", encoding="utf-8") as handle:
                    contents = handle.read()

                self.assertIn("SEED=", contents)
                self.assertIn("--seed ${SEED}", contents)
                self.assertIn("--deterministic 'True'", contents)


if __name__ == "__main__":
    unittest.main()
