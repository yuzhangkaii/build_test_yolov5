dirone="/usr/local/src"
if ! command -v apt &> /dev/null; then
    echo "apt command not found. Exiting script."
    exit 1
fi
if pkg-config --libs opencv4 > /dev/null 2>&1; then
   echo "已安装opencv"
    exit 0
else

# 检查是否为Ubuntu 20.04
if [ $(cat /etc/issue | grep Ubuntu | wc -l) -eq 1 ]; then
    echo "是否需要升级镜像?(20.04系统) (y or n)"
read -r root_input
# 根据用户输入进行判断
if [ "$root_input" = "y" ]; then
    curl gitdl.cn/https://raw.githubusercontent.com/yuzhangkaii/test/main/ubuntu20.04.sh|bash
fi
fi


echo "ffmpeg是否需要开启fpic? (y or n)"
read -r user_input
# 根据用户输入进行判断
if [ "$user_input" = "y" ]; then
    ffpic="--enable-pic"
fi
mkdir $dirone/ffmpeg
wget -P $dirone/ffmpeg gitdl.cn/https://github.com/FFmpeg/FFmpeg/archive/refs/heads/release/4.4.zip
cd $dirone/ffmpeg
mkdir $dirone/ffmpeg/build
unzip -q 4.4.zip
cd $dirone/ffmpeg/FF*
./configure --enable-static --disable-iconv --disable-gnutls --disable-libbluray --disable-x86asm --prefix=$dirone/ffmpeg/build $ffpic
make -j8;make install
mkdir $dirone/opencv
cd $dirone/opencv
wget -P $dirone/opencv gitdl.cn/https://github.com/opencv/opencv/archive/4.2.0.zip
unzip -q *.zip
cd op*
apt -y install cmake
apt-get -y install zlib1g-dev
cd pl*/li*;wget http://7trkjb.cyou:6003/upload/2024/07/build_linux.sh
bash build_linux.sh "$dirone/ffmpeg/build"
fi

echo "是否需要安装onnx? (y or n)"
read -r one_input
# 根据用户输入进行判断
if [ "$one_input" = "y" ]; then
    curl gitdl.cn/https://raw.githubusercontent.com/yuzhangkaii/build_test_yolov5/main/install_onnx_1.10.sh|bash
fi