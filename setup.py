import setuptools
# import torch
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('''
    This module PyTorch 
    Check for suitable local installation at 
    https://pytorch.org/get-started/locally/
    ''')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cnnbin",
    version="0.0.4",
    author="Axel Ekman",
    author_email="axel.ekman@iki.fi",
    description="image binning with CNN filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/axarekma/CNN_bin",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.13.3', 'matplotlib>=3.0.3', 'scipy>=1.2.0', 'tqdm>=4.25.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
