from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='drug_price_prediction',
    version='0.1',
    description='Drug Price Prediction Module',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    # Substitute <github_account> with the name of your GitHub account
    url='https://github.com/khalilouardini/drug_price_prediction',
    author='Khalil Ouardini',  # Substitute your name
    author_email='ouardini.k@gmail.com',  # Substitute your email
    license='MIT',
    packages=['drug_price_prediction'],
    install_requires=[
    'pypandoc>=1.7.2',
    'watermark>=2.2.0',
    'pandas>=1.3.5',
    'scikit-learn>=1.0.2',
    'scipy>=1.7.3',
    'matplotlib>=3.5.1',
    'pytest>=6.2.5',
    'pytest-runner>=5.3.1'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']  
)