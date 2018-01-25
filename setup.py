import os, shutil
# from Cython.Build import cythonize
from setuptools import setup

import elevation

pj = lambda *paths: os.path.abspath(os.path.join(*paths))
repo_dir = os.path.abspath(os.path.dirname(__file__))

# initialize settings.py file.
if not os.path.exists(pj(repo_dir, "elevation/settings.py")):
    print("Using default settings_template.py")
    shutil.copyfile("elevation/settings_template.py", "elevation/settings.py")


setup(
    name='Elevation',
    version=elevation.__version__,
    author='Nicolo Fusi and Jennifer Listgarten',
    author_email="fusi@microsoft.com, jennl@microsoft.com",
    description=("Machine Learning-Based Predictive Modelling of CRISPR/Cas9 off-target effects"),
    packages=["elevation"],
    install_requires=['scipy', 'numpy', 'matplotlib', 'nose', 'scikit-learn>=0.18', 'pandas', 'joblib', 'mock==2.0.0', 'multiprocess', 'statsmodels', 'requests', 'xlrd'],
    license="BSD",
    test_suite="tests",
    entry_points={
        'console_scripts': [
            'elevation-fixtures = elevation.cmds.fixtures:Fixtures.cli_execute',
            'elevation-fit = elevation.cmds.fit:Fit.cli_execute',
        ],
    }
)
