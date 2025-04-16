from setuptools import setup
import os
from glob import glob

package_name = 'my_shape_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',  # Replace with your name
    maintainer_email='user@example.com',  # Replace with your email
    description='ROS2 package for shape detection',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'shape_detection = my_shape_detector.shape_detection:main',
        ],
    },
)