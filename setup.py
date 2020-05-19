import setuptools

import recommender_metrics

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="recommender-metrics",
    version=recommender_metrics.__version__,
    author="Niall Twomey",
    author_email="twomeynj@gmail.com",
    description="Recommender metric evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niall-twomey/recommender_metrics",
    install_requires=["numpy", "scikit-learn", "tqdm"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",  # Hasn't been tested below this
    test_suite="nose.collector",
    tests_require=["nose"],
)
