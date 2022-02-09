from setuptools import setup, find_packages

install_requires = [
    "numpy>=1.15",
    "pymongo>=3.7",
    "pandas",
    "sacred>=0.8.1",
    "pyyaml>=5.1",
    "jsonpickle>=1.2, <2.0",
    "munch>=2.0.4",
    "tqdm>=4.36",
    "debugpy>=1.2.1"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='seml',
    version='0.3.6',
    description='Slurm Experiment Management Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/TUM-DAML/seml',
    author='DAML Group @ TUM',
    author_email='zuegnerd@in.tum.de, klicpera@in.tum.de',
    packages=find_packages('.'),
    include_package_data=True,
    entry_points={
            'console_scripts': [
                'seml = seml.main:main'
            ]
    },
    install_requires=install_requires,
    python_requires='>=3.7',
    zip_safe=False,
)
