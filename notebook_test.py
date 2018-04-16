"""

# Attribution

This Jupyter notebook testing code was found in the article
[Testing Jupyter Notebooks](https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/).

"""

import os
import subprocess
import tempfile

import nbformat


def _notebook_run(nbpath, timeout=60):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    nbpath = os.path.abspath(nbpath)

    os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.getcwd(), "src"))
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
        "jupyter", "nbconvert", "--to", "notebook", "--execute",
          "--ExecutePreprocessor.timeout={}".format(timeout),
          "--output", fout.name, nbpath]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
                     for output in cell["outputs"]
                     if output.output_type == "error"]


    return nb, errors


def test_autos():
    nb, errors = _notebook_run("notebooks/Autos2.ipynb")
    assert errors == []

def test_cyber():
    nb, errors = _notebook_run("notebooks/Cybercartography-mapalgo.ipynb", timeout=1200)
    assert errors == []
