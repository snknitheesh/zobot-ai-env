from setuptools import find_packages, setup

package_name = 'debug_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zozo',
    maintainer_email='snknitheesh@gmail.com',
    description='Python debug package for testing ROS2 setup',
    license='MIT',
    entry_points={
        'console_scripts': [
            'talker = debug_py.talker:main',
            'listener = debug_py.listener:main',
        ],
    },
)
