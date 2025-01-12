from setuptools import setup, find_packages

VERSION = '0.0.10'
DESCRIPTION = 'Training module for training PyTorch models'
LONG_DESCRIPTION = 'A versatile PyTorch training framework to simplify and enhance the model training process. It includes a trainer class with efficient training methods, famous built in pre-trained architectures, metrics tracking, custom and built-in callbacks support, and much more!'

# Setting up
setup(
    name="pytorch-candle",
    version=VERSION,
    author="Parag Londhe",
    author_email="<paraglondhe123@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url="https://github.com/paraglondhe/pytorch-candle",
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'matplotlib',
        'tqdm',
        'torchsummary'
    ],
    keywords=['python', 'pytorch', 'model training', 'deep learning', 'AI'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.7',
)