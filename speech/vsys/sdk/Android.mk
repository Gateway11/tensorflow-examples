LOCAL_PATH := $(call my-dir)

DEPENDENT_LIBRARIES_PATH := ../

include $(CLEAR_VARS)
LOCAL_PREBUILT_LIBS := libtensorflow_inference:libtensorflow_inference.so
include $(BUILD_MULTI_PREBUILT)

include $(CLEAR_VARS)
LOCAL_MODULE := libvoicerec_jni
LOCAL_SRC_FILES := com_vsys_voicerec_VoiceRecognize.cpp
LOCAL_C_INCLUDES :=
LOCAL_LDLIBS := -llog -L$(DEPENDENT_LIBRARIES_PATH)/libs/armeabi-v7a -lvad
LOCAL_CPPFLAGS := -std=c++11 -Wall -Wno-unused-parameter -I$(DEPENDENT_LIBRARIES_PATH)/api
LOCAL_SHARED_LIBRARIES :=
include $(BUILD_SHARED_LIBRARY)
