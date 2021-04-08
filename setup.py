from distutils.core import setup

setup(
    name = 'pipecaster',
    packages = ['pipecaster'],
    version = 'v0.3-alpha',  # same as your GitHub release tag varsion
    description = 'multichannel machine learning library with in-pipeline automation',
    long_description = 'Python library for building multichannel machine learning pipelines and for in-pipeline automation (semi-auto-ML)',
    author = 'A. John Callegari Jr.',
    author_email = 'a.john.callegari@gmail.com',
    license='MIT',
    url = 'https://github.com/ajcallegari/pipecaster',
    download_url = 'https://github.com/ajcallegari/pipecaster/archive/refs/tags/v0.3-alpha.tar.gz',
    keywords = ['machine learning', 'semi-auto-ml',
                'multichannel pipeline', 'workflow automation',
                'ensemble learning', 'machine learning pipeline'],
    classifiers = [],
)
