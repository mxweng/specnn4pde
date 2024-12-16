from distutils.command import install_data
from setuptools import setup, find_packages

setup(
    name='specnn4pde',  # package name
    version='0.1.1',  # version
    author='MXWeng',  # author name
    author_email='2431141461@qq.com',  # author email
    description='Solving partial differential equations using spectral methods and neural networks.',  # short description
    long_description=open('README.md').read(),  # long description, usually your README
    long_description_content_type='text/markdown',  # format of the long description, 'text/markdown' if it's Markdown
    url='https://github.com/mxweng/specnn4pde',  # project homepage
    packages=find_packages(),  # automatically discover all packages
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],  # list of classifiers
    python_requires='>=3.6',  # Python version requirement
    install_requires=['GPUtil',                      
                      'IPython',
                      'matplotlib',
                      'numpy', 
                      'pandas',
                      'Pillow',  # add Pillow for PIL               
                      'psutil', 
                      'PyPDF2>=3.0.0',
                      'scipy',
                      'sympy',
                      ],  # dependencies
)