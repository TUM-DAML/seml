from setuptools import setup, find_packages


setup(name='seml',
      version='0.1.0',
      description='Slurm Experiment Management Library',
      url='http://github.com/KDDgroup/seml',
      author='KDD Group @ TUM',
      author_email='zuegnerd@in.tum.de; klicpera@in.tum.de',
      packages=find_packages('.'),
      scripts=['bin/seml'],
      zip_safe=False)
