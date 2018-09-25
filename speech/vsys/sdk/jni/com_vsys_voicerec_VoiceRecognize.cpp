#include <jni.h>
#include <android/log.h>
#include <tensorflow/c/c_api.h>

#define LOG_TAG "VoiceRecognize.jni"
#define VSYS_DEBUGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define VSYS_DEBUGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define JNI_FUNCTION_DECLARATION(rettype, name, ...) \
    extern "C" JNIEXPORT rettype JNICALL Java_com_vsys_voicerec_VoiceRecognizeImpl_##name(__VA_ARGS__)

JavaVM* g_jvm = nullptr;
JNIEnv* env = nullptr;

TF_Status* status = nullptr;
TF_Graph* graph = nullptr;
TF_Session* session = nullptr;
std::vector<TF_Output> inputs, outputs;

JNI_FUNCTION_DECLARATION(jint, native_init, JNIEnv* env, jclass)
{
    // create graph
    graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_Buffer* graph_buf = TF_NewBufferFromString(data, size);
    TF_GraphImportGraphDef(graph, graph_buf, graph_opts, status);
    TF_DeleteBuffer(graph_buf);
    if (TF_GetCode(status) != TF_OK) {
        VSYS_DEBUGE("ERROR: Unable to import graph %s", TF_Message(status));
        return -1;
    }
    TF_DeleteImportGraphDefOptions(opts);

    // create session
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    session = TF_NewSession(graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        VSYS_DEBUGE("ERROR: Unable to create session %s", TF_Message(status));
        return -1;
    }
    TF_DeleteSessionOptions(session_opts);

    inputs.push_back({TF_GraphOperationByName(graph, "input_tensor"), 0});
    inputs.push_back({TF_GraphOperationByName(graph, "sequence_len"), 0});
    outputs.push_back({TF_GraphOperationByName(graph, "decode/CTCBeamSearchDecoder:1"), 0});
}

JNI_FUNCTION_DECLARATION(void, native_release, JNIEnv* env, jclass)
{
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}


JNI_FUNCTION_DECLARATION(jint, native_process, JNIEnv* env, jclass)
{
    std::vector<std::string> results;

    const int64_t tensor_dims[] = {1, 88, 494};
    TF_Tensor* input_tensor = TF_NewTensor(TF_UINT8, tensor_dims, 
            sizeof(tensor_dims), data, data_size, nullptr, nullptr);
    std::vector<TF_Tensor *> input_values{input_tensor};
    std::vector<TF_Tensor *> output_values(output_ops.size(), nullptr);

    TF_SessionRun(session, nullptr, &input_ops[0], &input_values[0], input_ops.size(), 
            &output_ops[0], &output_values[0], output_ops.size(), nullptr, 0, nullptr, status);
    if (TF_GetCode(status) != TF_OK) {
        VSYS_DEBUGE("ERROR: Unable to run session %s", TF_Message(status));
        return -1;
    }
    return 0;
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
