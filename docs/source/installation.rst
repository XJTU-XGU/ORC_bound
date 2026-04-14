Installation
===========

Requirements
------------

- Python >= 3.9
- numpy >= 1.21.0
- scipy >= 1.7.0
- networkx >= 3.0

Install from PyPI
-----------------

.. code-block:: bash

    pip install orc-bound

Install from source
-------------------

.. code-block:: bash

    git clone https://github.com/XJTU-XGU/ORC_bound.git
    cd ORC_bound
    pip install .

Install with development tools
------------------------------

.. code-block:: bash

    pip install -e ".[dev]"

This installs pytest, black, and ruff alongside the package.

Install with SerpAPI support
----------------------------

.. code-block:: bash

    pip install -e ".[serpapi]"

Required for :func:`orc_bound.utils.search.search_orc_literature`.
Obtain a free API key at https://serpapi.com/

Uninstall
---------

.. code-block:: bash

    pip uninstall orc-bound
