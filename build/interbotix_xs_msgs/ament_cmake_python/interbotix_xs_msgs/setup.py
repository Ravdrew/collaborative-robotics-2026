from setuptools import find_packages
from setuptools import setup

setup(
    name='interbotix_xs_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('interbotix_xs_msgs', 'interbotix_xs_msgs.*')),
)
