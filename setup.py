from __future__ import absolute_import
from setuptools import setup, find_packages


setup(name='PPN2V',
      version="0.1",
      description='Parametric Probabilistic Noise2Void (PPN2V) allows the training of a denoising CNN from individual noisy images with the help of a learnable noise model.'
                  'The noise model can take the form of a 2D histogram or a gaussian mixture model.',
      url='https://github.com/juglab/PPN2V/',
      author='Mangal Prakash, Manan Lalit, Pavel Tomancak, Alexander Krull, Florian Jug',
      author_email='TODO',
      license='BSD 3-Clause License',
      packages=find_packages(),
      project_urls={
          'Repository': 'https://github.com/juglab/PPN2V/',
      },
      install_requires=[
          "matplotlib",
          "numpy",
          "scipy",
          "tifffile",
          "torch",
          "torchvision"
      ]
      )
