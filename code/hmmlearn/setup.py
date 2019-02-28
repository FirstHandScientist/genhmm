# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2014 Gael Varoquaux
#               2014-2016 Sergei Lebedev <superbobry@gmail.com>
#               2018- Antony Lee


from distutils.version import LooseVersion
from io import open

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


# Added support for environment markers in install_requires.
if LooseVersion(setuptools.__version__) < "36.2":
    raise ImportError("setuptools>=36.2 is required")


class build_ext(build_ext, object):

    def finalize_options(self):
        # The key point: here, Cython and numpy will have been installed by
        # pip.
        from Cython.Build import cythonize
        import numpy as np
        import numpy.distutils

        self.distribution.ext_modules[:] = cythonize("**/*.pyx")
        # Sadly, this part needs to be done manually.
        for ext in self.distribution.ext_modules:
            for k, v in np.distutils.misc_util.get_info("npymath").items():
                setattr(ext, k, v)
            ext.include_dirs = [np.get_include()]

        super(build_ext, self).finalize_options()

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        super(build_ext, self).build_extensions()


setup(
    name="hmmlearn",
    description="",
    long_description=open("README.rst", encoding="utf-8").read(),
    maintainer="Antony Lee",
    maintainer_email="",
    url="https://github.com/hmmlearn/hmmlearn",
    license="new BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    cmdclass={"build_ext": build_ext},
    py_modules=[],
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    ext_modules=[Extension("", [])],
    package_data={},
    python_requires=">=2.7",
    setup_requires=[
        "Cython",
        "numpy",
        "setuptools_scm",
    ],
    use_scm_version=lambda: {  # xref __init__.py
        "version_scheme": "post-release",
        "local_scheme": "node-and-date",
        "write_to": "lib/hmmlearn/_version.py",
    },
    install_requires=[
        "numpy>=1.10",  # np.broadcast_to.
        "scikit-learn>=0.16",  # sklearn.utils.check_array.
    ],
    extras_require={
        "tests": ["pytest"],
        "docs": ["Sphinx", "sphinx-gallery", "Pillow", "matplotlib"],
    },
    entry_points={
        "console_scripts": [],
        "gui_scripts": [],
    },
)
