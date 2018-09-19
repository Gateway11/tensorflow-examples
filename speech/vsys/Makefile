CC := g++
LD := g++
NDK_BUILD := ndk-build
PROJECT_ROOT := $(shell pwd)
PLATFORM_SDK_VERSION ?= 23

voicerec.out-dir-objs := $(PROJECT_ROOT)/output/obj
voicerec.out-dir-libs := $(PROJECT_ROOT)/output/libs

voicerec.cpp.src-dirs := $(shell find cpp -maxdepth 3 -type d)
voicerec.cpp.src-files := $(foreach dir, $(voicerec.cpp.src-dirs), $(wildcard $(dir)/*.cpp))
voicerec.cpp.src-objs = $(patsubst %.cpp, %.o, $(voicerec.cpp.src-files))
voicerec.cpp.src-flags := -std=c++11 -fPIC #-Wno-format
voicerec.cpp.src-includes := -I./
voicerec.cpp.src-deps := -L./
voicerec.cpp.output-file := libvoicerec.so
voicerec.cpp.output-class = -shared

voicerec.ndk.app-script := $(PROJECT_ROOT)/example/android/Application.mk
voicerec.ndk.build-script := $(PROJECT_ROOT)/example/android/Android.mk

voicerec.java.src-dirs := $(shell find android/java -maxdepth 3 -type d)
voicerec.java.src-files := $(foreach dir, $(voicerec.java.src-dirs), $(wildcard $(dir)/*.java))
voicerec.java.sourcepath := example/android/java
voicerec.java.classpath := $(ANDROID_SDK_HOME)/platforms/android-$(PLATFORM_SDK_VERSION)/android.jar
voicerec.java.classes-outdir := $(PROJECT_ROOT)/output/classes
voicerec.java.output-file := $(voicerec.out-dir-libs)/voicerec.jar

.PHONY: all android clean test

$(voicerec.cpp.output-file): $(voicerec.cpp.src-objs)
	@mkdir -p $(voicerec.out-dir-libs)
	$(LD) $(voicerec.cpp.output-class) $^ $(voicerec.cpp.src-flags) $(voicerec.cpp.src-deps) -o $(voicerec.out-dir-libs)/$@

%.o:%.cpp
	$(CC) -c $< $(voicerec.cpp.src-flags) $(voicerec.cpp.src-includes) -o $@

all: $(voicerec.cpp.output-file) test

android: $(voicerec.java.output-file)
	@mkdir -p $(voicerec.out-dir-objs)
	@mkdir -p $(voicerec.out-dir-libs)
	$(NDK_BUILD) \
		NDK_PROJECT_PATH=$(PROJECT_ROOT) \
		NDK_APPLICATION_MK=$(voicerec.ndk.app-script) \
		APP_BUILD_SCRIPT=$(voicerec.ndk.build-script) \
		NDK_OUT=$(voicerec.out-dir-objs) \
		NDK_LIBS_OUT=$(voicerec.out-dir-libs) \
		PLATFORM_SDK_VERSION=$(PLATFORM_SDK_VERSION)

$(voicerec.java.output-file): $(voicerec.java.src-files)
ifeq ($(ANDROID_SDK_HOME),)
	$(error ANDROID_SDK_HOME missed, exec shell command 'export ANDROID_SDK_HOME=...')
endif
	mkdir -p $(voicerec.java.classes-outdir)
	mkdir -p $(voicerec.out-dir-libs)
	javac -sourcepath $(voicerec.java.sourcepath) -classpath $(voicerec.java.classpath) -d $(voicerec.java.classes-outdir) $(voicerec.java.src-files)
	jar cf $@ -C $(voicerec.java.classes-outdir) .

test: main.o
	$(CC) $^ $(voicerec.cpp.src-flags) $(voicerec.cpp.src-deps) -L$(voicerec.out-dir-libs) -lvoicerec -o test

clean:
	-rm -f *.o
	-rm -f cc/*.o
	-rm -f $(target)
	-rm -rf output
