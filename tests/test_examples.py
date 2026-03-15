import os
import importlib.util
import warnings

import pytest
from pytest_notebook.nb_regression import NBRegressionFixture, NBRegressionError
import nbformat

# Load docs/source/examples.py by file path (it's not an installed package).
_examples_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples.py"
)
_spec = importlib.util.spec_from_file_location("docs_examples", _examples_module_path)
_examples_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_examples_mod)

download_example_notebooks = _examples_mod.download_example_notebooks
get_sphinx_examples = _examples_mod.get_sphinx_examples

@pytest.mark.slow
def test_examples(subtests):

    # First download the example notebooks (if not already downloaded)
    examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_examples")
    download_example_notebooks(
        examples_path=examples_path,
        images_path=None, # Don't download images
    )

    # Get the list of example notebooks.
    notebooks = get_sphinx_examples()

    # Prepare the testing framework.
    fixture = NBRegressionFixture()

    for nb_name in notebooks:
        # The expected location of the notebook.
        nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", nb_name + ".ipynb")

        # We're also going to create a cleaned version of the notebook that removes
        # cells with the "slmsuite_experimental" metadata.
        nb_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", nb_name + "_clean.ipynb")
        nb_path3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", nb_name + "_run.ipynb")


        with subtests.test(f"testing {nb_name}"):
            # Check that the notebook file exists.
            assert os.path.isfile(nb_path), "Notebook not found."

            # Do surgery on the notebook to remove cells that contain
            # the "slmsuite_experimental" metadata, which is used to mark cells
            # that contain hardware that cannot be run in the testing environment.
            nb = None
            with open(nb_path, "r", encoding="utf8") as f:
                nb = nbformat.read(f, as_version=4)

            if nb is None:
                raise RuntimeError("Failed to read notebook.")
            else:
                nb.cells = [cell for cell in nb.cells if "slmsuite_experimental" not in cell.metadata.keys()]
                with open(nb_path2, "w", encoding="utf8") as f:
                    nbformat.write(nb, f)

                # Now run pytest on the notebook.
                try:
                    result = fixture.check(nb_path2)

                    # Save the final notebook with outputs for inspection
                    # (this doesn't seem to be working?).
                    with open(nb_path3, "w", encoding="utf8") as f:
                        nbformat.write(result.nb_final, f)
                except NBRegressionError as e:  # Ignore notebook diff errors (output changes).
                    warnings.warn(f"Notebook regression diff: {e}")
                except Exception as e:
                    raise