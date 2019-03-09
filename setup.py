from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.2.4',
                     'h5py==2.9.0',
                     'tensorflow==1.13.1',
                     'google-cloud-storage==1.14.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
