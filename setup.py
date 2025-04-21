from setuptools import setup, find_packages

setup(
    name='graph-analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchode>=0.2.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.2.0',
        'umap-learn>=0.5.3',
        'networkx>=3.0',
        'plotly>=5.14.0',
        'ipywidgets>=8.0.0',
        'tqdm>=4.65.0',
        'jupyter>=1.0.0',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for analyzing weighted directed graphs and visualizing dynamics',
    license='MIT',
    url='https://github.com/your-username/graph-analysis',
)