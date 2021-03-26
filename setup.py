from distutils.core import setup

setup(
    name = 'pipecaster',
    packages = ['pipecaster'],
    version = 'v0.1-alpha',  # Ideally should be same as your GitHub release tag varsion
    description = 'multichannel machine learning library with in-pipeline automation',
    author = 'A. John Callegari Jr.',
    author_email = 'a.john.callegari@gmail.com',
    license='MIT',
    url = 'https://github.com/ajcallegari/pipecaster/tree/master/pipecaster',
    download_url = 'https://github.com/ajcallegari/pipecaster/archive/refs/tags/v0.1-alpha.tar.gz',
    keywords = ['machine learning', 'semi-auto-ml',
                'multichannel pipeline', 'workflow automation',
                'ensemble learning', 'machine learning pipeline'],
    classifiers = [],
)
