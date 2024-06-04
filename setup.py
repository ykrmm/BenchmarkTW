from setuptools import find_packages, setup

with open('requirements.txt') as open_file:
    install_requires = open_file.read()
    
    
setup(
    name='tw_benchmark',
    version='1.0.0',
    packages=find_packages(),
    description='Benchmark Time Window',
    author='Yannis Karmim',
    license='MIT',
    author_email='yannis.karmim@cnam.fr',
    python_requires='>=3.7', 

)