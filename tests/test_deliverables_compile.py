"""Compile-check the SNL research scripts that the main unittest suite does not import."""
import pathlib
import py_compile
import unittest


class TestDeliverablesCompile(unittest.TestCase):
    def test_snl_causal_study_scripts_compile(self):
        study_dir = pathlib.Path(__file__).resolve().parents[1] / "deliverables" / "snl-causal-study"
        py_files = sorted(study_dir.glob("*.py"))
        self.assertTrue(py_files, "未找到 deliverables/snl-causal-study/*.py")
        for path in py_files:
            with self.subTest(file=path.name):
                py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    unittest.main()
