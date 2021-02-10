from setuptools import setup, find_packages

import versioneer


setup(
    name='openpifpaf_deeplabcut',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    license='MIT',
    description='OpenPifPaf plugin to read DeepLabCut data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/vita-epfl/openpifpaf_deeplabcut',

    install_requires=[
        'h5py',
        'openpifpaf>=0.12b2',
        'pyyaml',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
            'pycodestyle',
        ],
    },
)
