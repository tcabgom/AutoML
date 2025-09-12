from setuptools import setup, find_packages

setup(
    name='basicautoml',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        'scikit-learn>=1.0',
        'numpy>=1.21',
        'pandas>=1.3',
        'scipy>=1.7',
        'joblib>=1.0',
        'optuna>=2.10',
        'imbalanced-learn>=0.8'
    ],
)