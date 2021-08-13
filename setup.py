from setuptools import setup, find_packages

setup(name='torch-rl',
      version='9.0',
      description='A Basic setup for Reinforcement Learning using PyTorch',
      author='Rohit Gopalan',
      author_email='rohitgopalan1990@gmail.com',
      license='DST',
      packages=find_packages(),
      install_requires=['numpy', 'torch', 'gym', 'scikit-learn'],
      zip_safe=False)
