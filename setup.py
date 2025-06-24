from setuptools import setup, find_packages

setup(
    name='scCulturePrec',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'h5py',
        'scikit-learn',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'scCulturePrec = scCulturePrec.runner:main',
        ],
    },
)

