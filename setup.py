from setuptools import find_packages, setup

setup(
    name="mypackage",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
        'importlib-metadata; python_version>="3.10"',
        "tarfile",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
)
