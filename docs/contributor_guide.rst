Contributing
============

As you code
-----------

1. Keep changes focused and use descriptive names.
2. Add Google-style docstrings to public APIs so Sphinx can include them in the
   :doc:`API reference <api_reference>`.
3. Add tests for new behavior and regressions.
4. Run the formatter, linter, and relevant tests before requesting review.

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

1. Create a short-lived branch from ``main``.
2. Open a draft pull request early when feedback would help.
3. Explain why the change is needed, how it was tested, and any GPU or data
   requirements needed to reproduce it.
4. Keep the review surface small and separate unrelated changes.

As you review
-------------

Review for scientific correctness, reproducibility, clear configuration
behavior, and appropriate tests. Be direct about technical risks while keeping
feedback constructive and actionable.

Contributing to the documentation
---------------------------------

The documentation is built with `Sphinx
<https://www.sphinx-doc.org/en/master/>`_, uses the `PyData Sphinx Theme
<https://pydata-sphinx-theme.readthedocs.io/>`_, and is hosted on GitHub Pages.
Narrative pages may be written in reStructuredText or Markdown through MyST.
Runnable gallery examples live in ``docs/examples/``.

Install the documentation dependencies:

.. code-block:: bash

   uv pip install -r docs/docs_requirements.txt

Build the site from the repository root:

.. code-block:: bash

   make -C docs html

The generated site is written to ``docs/_build/html/``. Treat Sphinx warnings
as issues to resolve before merging, especially broken references and failed
autodoc imports.

Documentation structure
~~~~~~~~~~~~~~~~~~~~~~~

* ``docs/tutorial/`` — guided first-use workflows
* ``docs/guides/`` — task-oriented Markdown guides
* ``docs/api/`` — API pages generated from Python docstrings
* ``docs/examples/`` — runnable tutorial notebooks rendered with ``nbgallery``
* ``docs/_static/`` — logos and theme overrides
