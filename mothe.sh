apt-get remove x264 libx264-dev -y

apt-get install build-essential checkinstall cmake pkg-config yasm -y
apt-get install git gfortran -y
apt-get install libjpeg8-dev libjasper-dev libpng12-dev -y

apt-get install libtiff5-dev -y

apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev -y
apt-get install libxine2-dev libv4l-dev -y
apt-get install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev -y
apt-get install qt5-default libgtk2.0-dev libtbb-dev -y
apt-get install libatlas-base-dev -y
apt-get install libfaac-dev libmp3lame-dev libtheora-dev -y
apt-get install libvorbis-dev libxvidcore-dev -y
apt-get install libopencore-amrnb-dev libopencore-amrwb-dev -y
apt-get install x264 v4l-utils -y


apt-get install libprotobuf-dev protobuf-compiler -y
apt-get install libgoogle-glog-dev libgflags-dev -y
apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen -y

apt-get install python-dev python-pip python3-dev python3-pip -y
pip3 install -U pip numpy

pip3 install virtualenv virtualenvwrapper
echo "# Virtual Environment Wrapper"  >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc


