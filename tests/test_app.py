import subprocess
import sys
import unittest


class AppSmokeTest(unittest.TestCase):
    def test_app_module_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "app"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "SmartGroceryAI")


if __name__ == "__main__":
    unittest.main()
