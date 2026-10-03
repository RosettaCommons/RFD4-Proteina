Contributing
============

As you code
-----------

1. Keep changes focused and use descriptive names.
2. Add Google style docstrings to public APIs so Sphinx can pull them into the
   :doc:`API reference <api_reference>`.
3. Add tests for new behavior and for regressions.
4. Run the formatter, the linter, and the relevant tests before you request a
   review.

.. code-block:: bash

   make format
   ruff check src tests
   pytest -m "not gpu"

As you commit
-------------

* Keep each commit to one logical change.
* Use `Conventional Commits
  <https://www.conventionalcommits.org/en/v1.0.0/>`_, for example
  ``docs: explain indexed motif conditioning``.
* Do not commit checkpoints, generated designs, credentials, or local
  environment files.

As you open a pull request
--------------------------

1. Create a short lived branch from ``main``.
2. Open a draft pull request early when feedback would help.
3. Explain why the change is needed, how you tested it, and what GPU or data
   someone needs to reproduce it.
4. Keep the review surface small, and split unrelated changes apart.

As you review
-------------

Review for scientific correctness, reproducibility, clear configuration
behavior, and appropriate tests. Be direct about technical risk, and keep the
feedback something the author can act on.

Contributing to the documentation
---------------------------------

We build the documentation with `Sphinx
<https://www.sphinx-doc.org/en/master/>`_ and the `PyData Sphinx Theme
<https://pydata-sphinx-theme.readthedocs.io/>`_, and host it on GitHub Pages.
Write narrative pages in either reStructuredText or Markdown through MyST.
Runnable gallery examples live in ``docs/examples/``.

Create the environment without downloading model weights, then install the
documentation dependencies. Building documentation does not require a GPU:

.. code-block:: bash

   bash setup.sh --skip-weights
   uv pip install --python .venv/bin/python -r docs/docs_requirements.txt

Build the site from the repository root:

.. code-block:: bash

   make -C docs html SPHINXBUILD="$PWD/.venv/bin/sphinx-build" SPHINXOPTS="-W --keep-going"

The generated site lands in ``docs/_build/html/``. Treat every Sphinx warning as
something to fix before merging, especially broken references and failed autodoc
imports.

Documentation structure
~~~~~~~~~~~~~~~~~~~~~~~

* ``docs/quickstart.rst`` covers install and a first design
* ``docs/workflow/`` holds one page per stage of a run
* ``docs/reference/`` holds lookup tables and syntax
* ``docs/api/`` holds API pages generated from Python docstrings
* ``docs/examples/`` holds runnable tutorial notebooks rendered with ``nbgallery``
* ``docs/_static/`` holds logos and theme overrides
