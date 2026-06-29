import platform
import sys

from setuptools import Extension, find_packages, setup

try:
    import pybind11
except ImportError:
    print(
        "pybind11 is required to build orc_bound. "
        "Install it with: python -m pip install pybind11",
        file=sys.stderr,
    )
    raise


def compiler_args():
    system = platform.system()
    if system == "Windows":
        return ["/O2", "/openmp", "/std:c++17"], []
    if system == "Darwin":
        # macOS Clang does not ship OpenMP by default. The extension still
        # builds and runs single-threaded unless OpenMP flags are added by users.
        return ["-O3", "-std=c++17"], []
    return ["-O3", "-std=c++17", "-fopenmp"], ["-fopenmp"]


extra_compile_args, extra_link_args = compiler_args()

with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    name="orc_bound",
    version="0.2.1",
    description="C++/OpenMP accelerated Ollivier-Ricci curvature bounds for NetworkX graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XJTU-XGU/ORC_bound",
    project_urls={
        "Homepage": "https://github.com/XJTU-XGU/ORC_bound",
        "Repository": "https://github.com/XJTU-XGU/ORC_bound",
    },
    python_requires=">=3.7",
    install_requires=[
        "networkx>=2.6",
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    keywords=[
        "ollivier-ricci",
        "curvature",
        "graph",
        "optimal-transport",
        "openmp",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[
        Extension(
            "orc_bound._residual_shell_cpp",
            ["src/orc_bound/_residual_shell_cpp.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
