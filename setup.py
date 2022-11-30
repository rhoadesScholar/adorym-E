import setuptools

setuptools.setup(
    name='adorym-E',
    version='0.1.0',
    author='Jeff Rhoades',
    description='Automatic differentiation-based object retrieval with dynamic modeling, by Ming Du - Enhanced by Jeff Rhoades',
    packages=setuptools.find_packages(exclude=['docs']),
    include_package_data=True,
    url='https://github.com/rhoadesScholar/adorym-E.git',
    keywords=['adorym-E', 'adorym', 'automatic differentiation', 'object retrieval', 'dynamic modeling'],
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers'
    ]
)
