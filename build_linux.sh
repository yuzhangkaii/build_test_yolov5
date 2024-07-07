#!/bin/bash
BUILD_TYPE='Release'

rm -rf build_linux

mkdir build_linux
cd build_linux

cmake ../../../  \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_world=OFF \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_JAVA=OFF \
    -DBUILD_ANDROID_EXAMPLES=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_IPP=OFF \
    -DWITH_TBB=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DWITH_GTK=ON \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_FFMPEG=ON

make -j4
