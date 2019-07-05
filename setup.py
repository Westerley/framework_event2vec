try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='event2vec++',
    version='0.0.1',
    description='',
    author='Westerley Reis',
    author_email='',
    license='MIT',
    url=''
    install_requires=[
        'wheel',
        'Cython',
        'argparse',
        'futures',
        'six',
        'psutil'
        'networkx',
        'gensim',
        'numpy',
        'pandas',
        'scipy',
        'twdm',
        'joblib',
        'theano',
        'tensorflow',
    ],
    keywords=['machine learning', 'embeddings', 'event2vec', 'node2vec', 'deepwalk', 'line', 'metapath2vec', 'netmf']
)