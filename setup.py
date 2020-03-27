import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="recommender-metrics-nialltwomey",
    version="0.1.0",
    author="Niall Twomey",
    author_email="twomeynj@gmail.com",
    description="A basic library with implementations of some useful recommendation metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niall-twomey/recommender_metrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
