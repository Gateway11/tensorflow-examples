LOCAL_PATH := $(call my-dir)

DEPENDENT_LIBRARIES_PATH := $(LOCAL_PATH)/thirdparty/libs/$(TARGET_ARCH_ABI)

include $(CLEAR_VARS)
LOCAL_MODULE := libtensorflow_inference
LOCAL_SRC_FILES := $(DEPENDENT_LIBRARIES_PATH)/libtensorflow_inference.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libztvad
LOCAL_SRC_FILES := $(DEPENDENT_LIBRARIES_PATH)/libztvad.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libr2ssp
LOCAL_SRC_FILES := $(DEPENDENT_LIBRARIES_PATH)/libr2ssp.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libvoicerec_jni
LOCAL_SRC_FILES := com_tensorflow_voicerec_example_MainActivity.cpp
LOCAL_C_INCLUDES := $(LOCAL_PATH)/thirdparty/include
LOCAL_LDLIBS := -llog -L$(DEPENDENT_LIBRARIES_PATH) -lfftw3f
LOCAL_CPPFLAGS := -std=c++11 -Wall -Wno-unused-parameter -I$(LOCAL_PATH)/../../../../../../tensorflow
LOCAL_SHARED_LIBRARIES := libtensorflow_inference libztvad libr2ssp
include $(BUILD_SHARED_LIBRARY)
