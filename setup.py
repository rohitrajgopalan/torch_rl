from setuptools import setup, find_packages

setup(name='torch-rl',
      version='2.0',
      description='A Basic setup for Reinforcement Learning using PyTorch',
      author='Rohit Gopalan',
      author_email='rohitgopalan1990@gmail.com',
      license='DST',
      packages=find_packages(),
      install_requires=['numpy', 'torch', 'gym'],
      zip_safe=False)
