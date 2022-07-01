from setuptools import setup

setup(
    name='batch_processing_discovery',
    version='0.1.0',
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'wittgenstein'
    ],
)
