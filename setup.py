from setuptools import setup
from setuptools import find_packages


setup(
    name='tgen',
    version='0.3.0',
    description='Sequence-to-sequence natural language generator',
    author='Ondrej Dusek',
    author_email='odusek@ufal.mff.cuni.cz',
    url='https://github.com/UFAL-DSG/tgen',
    download_url='https://github.com/UFAL-DSG/tgen.git',
    license='Apache 2.0',
    install_requires=['regex',
                      'unicodecsv',
                      'enum34',
                      'numpy',
                      'rpyc',
                      'pudb',
                      'recordclass',
                      'tensorflow==1.15.0',
                      'kenlm',
                      'pytreex==0.1dev'],
    dependency_links=['https://github.com/kpu/kenlm/archive/master.zip#egg=kenlm',
                      'https://github.com/ufal/pytreex/tarball/master#egg=pytreex-0.1dev'],
    packages=find_packages()
)

