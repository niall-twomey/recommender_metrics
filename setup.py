import setuptools

from recommender_metrics import __version__ as version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="recommender-metrics",
    version=version,
    author="Niall Twomey",
    author_email="twomeynj@gmail.com",
    description="Recommender metric evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niall-twomey/recommender_metrics",
    install_requires=[
        'pandas',
        'scikit-learn',
        'tqdm',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)