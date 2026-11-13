from setuptools import setup

package_name = 'floor_inspect'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/inspect.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lacrep',
    maintainer_email='noreply@example.com',
    description='Floor unevenness inspection (potholes/bumps/lips) for AMRs using RealSense D455',
    license='MIT',
    entry_points={
        'console_scripts': [
            'inspect_node = floor_inspect.inspect_node:main',
            'floor_inspect = floor_inspect.cli:main',
        ],
    },
)
