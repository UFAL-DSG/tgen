from setuptools import setup
from setuptools import find_packages


setup(
    name='tgen',
    version='0.2.0',
    description='Sequence-to-sequence natural language generator',
    author='Ondrej Dusek',
    author_email='odusek@ufal.mff.cuni.cz',
    url='https://github.com/UFAL-DSG/tgen',
    download_url='https://github.com/UFAL-DSG/tgen.git',
    license='Apache 2.0',
    packages=find_packages()
)

