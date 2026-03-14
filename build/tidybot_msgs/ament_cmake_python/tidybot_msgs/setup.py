from setuptools import find_packages
from setuptools import setup

setup(
    name='tidybot_msgs',
    version='0.1.0',
    packages=find_packages(
        include=('tidybot_msgs', 'tidybot_msgs.*')),
)
