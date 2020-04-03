from setuptools import setup, find_packages


setup(
    name='seml',
    version='0.1.0',
    description='Slurm Experiment Management Library',
    url='http://github.com/TUM-DAML/seml',
    author='DAML Group @ TUM',
    author_email='zuegnerd@in.tum.de; klicpera@in.tum.de',
    packages=find_packages('.'),
    entry_points={
            'console_scripts': [
                'seml = seml.main:main'
            ]
    },
    zip_safe=False)
