#!/bin/bash

FFTW_VERSION=fftw-3.3.8
INSTALL_DIR=`pwd`

wget ftp://ftp.fftw.org/pub/fftw/${FFTW_VERSION}.tar.gz
tar zxf ${FFTW_VERSION}.tar.gz

export PATH="$NDK_ROOT/toolchains/arm-linux-androideabi-4.8/prebuilt/linux-x86_64/bin/:$PATH"
export CC="arm-linux-androideabi-gcc --sysroot=$NDK_ROOT/platforms/android-19/arch-arm/ -march=armv7-a -mfloat-abi=softfp"
export LD="arm-linux-androideabi-ld"
export AR="arm-linux-androideabi-ar"
export RANLIB="arm-linux-androideabi-ranlib"
export STRIP="arm-linux-androideabi-strip"
#export CFLAGS="-mfpu=neon -mfloat-abi=softfp"

cd $FFTW_VERSION
./configure --host=arm-eabi \
	--prefix=$INSTALL_DIR \
	LIBS="-lc -lgcc" \
	--enable-float \
	--enable-threads \
#	--with-combined-threads \
	--enable-neon

make -j4
make install
