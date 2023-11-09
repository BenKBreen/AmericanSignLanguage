from setuptools import setup, find_packages
setup(
   name='American Sign Language Reader',
   version='1.0',
   author='Benjamin Breen',
   author_email='benjaminkbreen@gmail.com',
   description='Processes a video to interpret 59 distinct symbols in American Sign Language',
   packages=find_packages(),
   entry_points={
      'console_scripts': [
         'aslread=asl.command_line:aslread',
      ],
   },
)
