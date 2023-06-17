#!/usr/bin/env bash

TENSORFLOW_VERSION=master
NDK_HOME=/Users/daixiang/Home/toolbox/ndk-r21

if [ ! -d tensorflow ]; then
    git clone --recurse-submodules https://github.com/tensorflow/tensorflow --depth=1 -b $TENSORFLOW_VERSION || exit 1;
    cd tensorflow/
else
    cd tensorflow/
    bazel clean &>/dev/null 

    # clean temp branch
    git reset --hard
    git clean -f -d
    git checkout $TENSORFLOW_VERSION
    git branch -D __temp__
fi

# creates a temp branch for apply some patches and reuse cloned folder
git checkout -b __temp__

# sets git local config for apply patch
git config user.email "temp@example.com"
git config user.name "temp"

yes '' | ./configure || {
#    log_failure_msg "error when configure tensorflow"
    exit 1
}

# Add a new tensorflow target bundling the C API over the Android specific TF core lib
cat << EOF >> WORKSPACE
android_ndk_repository(
    name="androidndk",
    path="$NDK_HOME",
    api_level=23)
EOF

cd tensorflow/tools/android/inference_interface
sed -i '' s/libtensorflow_inference.so/libtensorflow_inference2.so/g BUILD || exit 1;
cd ../../../../
#sed -i '' '/-Wl,--gc-sections/i\
#  "-Wl,-soname=libtensorflow_inference.so",' tensorflow/tools/android/inference_interface/BUILD

cat << EOF >> tensorflow/tools/android/inference_interface/BUILD
cc_binary(
    name = "libtensorflow_inference.so",
    srcs = [],
    copts = tf_copts() + [
        "-ffunction-sections",
        "-fdata-sections",
    ],
    linkopts = if_android([
        "-landroid",
        "-latomic",
        "-ldl",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--gc-sections",
	      # soname is required for the so to load on api > 22
        "-Wl,-soname=libtensorflow_inference.so",
        "-Wl,--version-script",
        "//tensorflow/c:version_script.lds",
    ]),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:version_script.lds",
        "//tensorflow/core:android_tensorflow_lib",
    ],
)
EOF

git add .
git commit -m "temp modifications"

bazel build -c opt //tensorflow/tools/android/inference_interface:libtensorflow_inference.so \
  --verbose_failures \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=armeabi-v7a

TF_SO="`pwd`/bazel-bin/tensorflow/tools/android/inference_interface/libtensorflow_inference.so"
TF_HEADER="`pwd`/tensorflow/c/c_api.h"
