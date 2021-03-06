# Holodeck-Time-Travel-Task-Analytics
This project contains the analytics and visualization code for the Holodeck-Time-Travel-Task project.

It is assumed you know how to get access to data (not included here) in order to run this analysis. In the future, sample data may be provided. For now, e-mail kevin.horecka@gmail.com to get access to sample data.

#Setup/Install

To use this program, the following software should be installed. This has only been tested on Windows 10, but other version of Windows should work (and perhaps other OSes as nothing is particularly OS specific).

1. Install Anaconda Python 2.7 64bit (32 might work, haven't tested) from https://www.continuum.io/downloads
2. Install Microsoft Visual C++ Compiler for Python 2.7 from http://aka.ms/vcpython27
3. Run the following command:
    
    `pip install PySide pyqtgraph colour tzlocal PyOpenGL PyOpenGL_accelerate argparse`
4. (optional) Install PyCharm Community Edition from https://www.jetbrains.com/pycharm/download/
5. Download this repo as zip (https://github.com/kevroy314/Holodeck-Time-Travel-Task-Analytics/archive/master.zip) and unzip in a known location or call git clone
6. Either open Main.py or PathVisualizerAnimated3D.py in PyCharm and click the Green Run Arrow or navigate to the unzipped path in a command prompt window and call "python Main.py" or "python PathVisualizerAnimated3D.py".

#Keyboard Commands

Space - Pause animation

\+ (plus arrow) - Speed up animation

\- (minus arrow) - Slow down animation

r - Reset timeline

e - Go to end of timeline

Escape - Exit the program

1 - Hide/show grid lines

2 - Hide/show base image

3 - Hide/show color bars

4 - Hide/show items

5 - Hide/show path line
