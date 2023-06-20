from setuptools import setup, find_packages

install_requires = [
    "numpy>=1.15",
    "pymongo>=3.11",
    "pandas",
    "sacred>=0.8.4",
    "pyyaml>=5.1",
    "jsonpickle>=2.2",
    "munch>=2.0.4",
    "tqdm>=4.36",
    "debugpy>=1.2.1",
    "requests>=2.28.1",
    "argcomplete",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='seml',
    version='0.3.7',
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
    python_requires='>=3.8',
    zip_safe=False,
)
