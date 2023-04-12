from setuptools import setup

setup(name='batchviewer',
      version='0.1',
      description='Simple but effective tool to visualize 3D data (with color channels)',
      url='https://github.com/FabianIsensee/BatchViewer.git',
      author='Fabian Isensee',
      author_email='isenseef@gmail.com',
      packages=['batchviewer'],
      zip_safe=False,
      install_requires=[
            'pyqtgraph',
            'pyqt5'
      ])
