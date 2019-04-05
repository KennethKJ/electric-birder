from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=1.3']

setup(
    name='electric-birder',
    version='0.1',
    author='Kenneth Kragh Jensen',
    author_email='kenneth.kragh.jensen@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Baby Weight prediction in Cloud ML',
    requires=[]
)
