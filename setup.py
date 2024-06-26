from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(name="idil",
      version="0.0.1",
      author="Sangwon Seo",
      author_email="sangwon.seo@rice.edu",
      description="Intent-Driven Imitation Learning",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(exclude=["apps"]),
      python_requires='>=3.8',
      install_requires=[
          'numpy',
          'click',
          'tqdm',
          'scipy',
          'sparse',
          'torch',
          'termcolor',
          'tensorboard',
          'gym==0.21.0',
          'mujoco-py<2.2,>=2.1',
          "stable-baselines3<=1.8.0,>=1.1.0",
          'cython<3',
          'opencv-python',
          'Box2D',
          'hydra-core>=1.3',
          'wandb>=0.15',
          'imitation==0.4',
          'pip==21',
          'setuptools==65.5.0 ',
          'swig',
          'wheel==0.38.0',
          'python-dotenv'
      ])
