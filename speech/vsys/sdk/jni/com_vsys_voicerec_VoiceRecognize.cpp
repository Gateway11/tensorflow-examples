#include <jni.h>
#include <android/log.h>

#include "c_api.h"

#define LOG_TAG "VoiceRecognize.jni"
#define VSYS_DEBUGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define VSYS_DEBUGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define JNI_FUNCTION_DECLARATION(rettype, name, ...) \
    extern "C" JNIEXPORT rettype JNICALL Java_com_vsys_voicerec_VoiceRecognizeImpl_##name(__VA_ARGS__)

JavaVM* g_jvm = nullptr;
JNIEnv* env = nullptr;

JNI_FUNCTION_DECLARATION(void, native_test, JNIEnv* env, jclass)
{

}

jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    g_jvm = vm;
    JNIEnv* env = nullptr;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_4) != JNI_OK) {
        return -1;
    }
    return JNI_VERSION_1_4;
}
