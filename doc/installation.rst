Installation
============

Requirements
------------

``orc_bound`` requires:

* Python 3.7 or newer
* NetworkX
* NumPy
* SciPy
* a C++17 compiler when building from source
* pybind11 during extension builds

Install from PyPI:

.. code-block:: bash

    python -m pip install --upgrade pip
    python -m pip install orc-bound

Source Builds
-------------

If a wheel is not available for your platform, pip builds the C++ extension
locally.

On Linux:

.. code-block:: bash

    sudo apt update
    sudo apt install -y build-essential python3-dev
    python -m pip install --upgrade pip setuptools wheel pybind11
    python -m pip install orc-bound

On Windows, install Visual Studio Build Tools with the C++ workload, then run:

.. code-block:: powershell

    python -m pip install --upgrade pip setuptools wheel pybind11
    python -m pip install orc-bound

On macOS:

.. code-block:: bash

    xcode-select --install
    python -m pip install --upgrade pip setuptools wheel pybind11
    python -m pip install orc-bound

The default macOS compiler setup usually does not include OpenMP, so the C++
kernel may run single-threaded unless you configure an OpenMP-capable compiler.

Building the Documentation
--------------------------

The repository includes a ``doc`` directory and a Read the Docs v2
configuration file. To build locally:

.. code-block:: bash

    python -m pip install -r doc/requirements.txt
    python -m sphinx -b html doc doc/_build/html

Read the Docs should use ``.readthedocs.yaml`` at the repository root. If this
package is used inside a monorepo, configure the Read the Docs project to use
this package directory or move the configuration file to the Git repository
root and keep the paths relative to that root.
