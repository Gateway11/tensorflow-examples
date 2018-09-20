#!/bin/sh

NDK_DIR=/Users/gaopeng/Documents/Libs/android-ndk-r10e
INSTALL_DIR=/Users/gaopeng/Documents/Libs/blis/blis-armv7a-mt
SRC_DIR=/Users/gaopeng/Documents/Libs/blis

cd $SRC_DIR


export PATH=$NDK_DIR/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/bin:$PATH
export SYS_ROOT="$NDK_DIR/platforms/android-19/arch-arm/"
export C_INCLUDE_PATH=$SYS_ROOT/usr/include
export CC="arm-linux-androideabi-gcc --sysroot=$SYS_ROOT -march=armv7-a -mfloat-abi=softfp"
export LD="arm-linux-androideabi-ld"
export AR="arm-linux-androideabi-ar"
export RANLIB="arm-linux-androideabi-ranlib"
export STRIP="arm-linux-androideabi-strip"
#export CFLAGS="-mfpu=neon -mfloat-abi=softfp"
export BLIS_ENABLE_VERBOSE_MAKE_OUTPUT=yes

mkdir -p $INSTALL_DIR
./configure -p $INSTALL_DIR armv7a

make clean
make -j4
make install

exit 0
