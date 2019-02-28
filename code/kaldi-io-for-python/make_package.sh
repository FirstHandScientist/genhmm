#!/bin/bash

exit 0

# Step 1: udpate version in 'setup.py'

# Step 2: make packages,
# python3 -m pip install --user --upgrade setuptools wheel twine
rm dist/*
python2 setup.py bdist_wheel
python3 setup.py sdist bdist_wheel
#$ ls dist/
#    kaldi_io-vesis84-0.9.0.tar.gz
#    kaldi_io_vesis84-0.9.0-py2-none-any.whl
#    kaldi_io_vesis84-0.9.0-py3-none-any.whl

# Hint: skip to 'Step 8' to skip sandboxing on 'test.pypi.org'

{ # TEST_DEPLOYMENT_ON test.pypi.org,
  # Step 3: upload the packages (test site),
  python3 -m twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

  # Step 4: see webpage,
  # https://test.pypi.org/project/kaldi_io_vesis84

  # Step 5: try installing it locally,
  python3 -m pip install --user --index-url https://test.pypi.org/simple/ kaldi_io_vesis84

  # Stepy 6: try to install it,
  python3
  <import kaldi_io
  <print(kaldi_io)

  # Step 7: remove the package,
  python3 -m pip uninstall kaldi_io_vesis84
}

# Step 8: Put the packages to 'production' pypi,
python3 -m twine upload --verbose dist/*
python3 -m pip install --user kaldi_io
python3 -m pip uninstall kaldi_io

