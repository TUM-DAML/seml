from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='seml',
    version='0.1.2',
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
    install_requires=install_requires,
    zip_safe=False)
