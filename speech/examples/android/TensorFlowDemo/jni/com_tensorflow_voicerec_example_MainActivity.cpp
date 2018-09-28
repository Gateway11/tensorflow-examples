#include <stdlib.h>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <android/log.h>
#include <jni.h>
#include <fftw3.h>
#include <NNVadIntf.h>
#include <tensorflow/c/c_api.h>
#include "buf_manager.h"
#include "utils.h"

#define NUM_INPUTS 2
#define NUM_OUTPUTS 1
#define FRAME_SIZE 160

#define LOG_TAG "VoiceRecognize.jni"
#define VSYS_DEBUGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define VSYS_DEBUGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define JNI_FUNCTION_DECLARATION(rettype, name, ...) \
    extern "C" JNIEXPORT rettype JNICALL Java_com_tensorflow_voicerec_example_MainActivity_##name(__VA_ARGS__)
JavaVM* g_jvm = nullptr;
JNIEnv* env = nullptr;

TF_Status* status = nullptr;
TF_Graph* graph = nullptr;
TF_Session* session = nullptr;
TF_Output inputs[NUM_INPUTS];
TF_Output outputs[NUM_OUTPUTS];
VD_HANDLE vad_handle;

std::map<uint32_t, std::string> symbol_table;
uint32_t vad_input_offset = 0;
uint32_t vad_input_total = 0;
float* vad_input = nullptr;

JNI_FUNCTION_DECLARATION(jint, native_init, JNIEnv* env, jclass, jstring base_path)
{
    const char* path = env->GetStringUTFChars(base_path, NULL);
    std::string nnet_str = read(std::string(path) + "/output_graph.pb");
    read_symbol_table(std::string(path) + "/symbol_table.txt");
    env->ReleaseStringUTFChars(base_path, path);
    // create graph
    graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_Buffer* graph_buf = TF_NewBufferFromString(nnet_str.c_str(), nnet_str.size());
    TF_GraphImportGraphDef(graph, graph_buf, graph_opts, status);
    TF_DeleteBuffer(graph_buf);
    if (TF_GetCode(status) != TF_OK) {
        VSYS_DEBUGE("ERROR: Unable to import graph %s", TF_Message(status));
        return -1;
    }
    TF_DeleteImportGraphDefOptions(graph_opts);
    // create session
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    session = TF_NewSession(graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        VSYS_DEBUGE("ERROR: Unable to create session %s", TF_Message(status));
        return -1;
    }
    TF_DeleteSessionOptions(session_opts);

    inputs[0] = TF_Output{TF_GraphOperationByName(graph, "input_tensor"), 0};
    inputs[1] = TF_Output{TF_GraphOperationByName(graph, "sequence_len"), 0};
    outputs[0] = TF_Output{TF_GraphOperationByName(graph, "decode/CTCBeamSearchDecoder:1"), 0};

    vad_handle = VD_NewVad(1);
}

JNI_FUNCTION_DECLARATION(void, native_release, JNIEnv* env, jclass)
{
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    VD_DelVad(vad_handle);
}

JNI_FUNCTION_DECLARATION(jint, native_process, JNIEnv* env, jclass, jbyteArray input, jint length)
{
	jbyte* data = env->GetByteArrayElements(input, NULL);
	env->ReleaseByteArrayElements(input, data, JNI_ABORT);

    std::vector<std::string> results;
    const int64_t input_dims[] = {1, 88, 494};
    TF_Tensor* input_values[NUM_INPUTS];
    TF_Tensor* output_values[NUM_OUTPUTS];

//    input_values[0] = TF_NewTensor(TF_UINT8, input_dims, 
//            sizeof(input_dims), data, data_size, nullptr, nullptr);
//    input_values[1] = TF_NewTensor(TF_UINT8, input_dims, 
//            sizeof(input_dims), data, data_size, nullptr, nullptr);
//    output_values[0] = TF_NewTensor(TF_UINT8, input_dims, 
//            sizeof(input_dims), data, data_size, nullptr, nullptr);
//    TF_SessionRun(session, nullptr, inputs, &input_values, NUM_INPUTS, 
//            outputs, output_values, NUM_OUTPUTS, nullptr, 0, nullptr, status);
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

int32_t check_voice_activity(const float* input, const uint32_t input_size){
    uint32_t num_frames = (input_size + vad_input_offset) / FRAME_SIZE;
    write_to_buffer(&vad_input, &vad_input_offset, &vad_input_total, input, input_size);
    
    uint32_t i, ret = 0;
    for (i = 0; i < num_frames; i++) {
        VD_InputFloatWave(vad_handle, vad_input + i * FRAME_SIZE, FRAME_SIZE, false, false);
        if(!has_vad){
            if(VD_GetVoiceStartFrame(vad_handle) >= 0) has_vad = true;
        }else{
            if(VD_GetVoiceStopFrame(vad_handle)){
                VD_RestartVad(vad_handle);
                ret = 1;
                has_vad = false;
                i++;
                break;
            }
        }
    }
    vad_input_offset -= i * FRAME_SIZE;
    memcpy(vad_input, vad_input + i * num_frames, vad_input_offset * sizeof(float));
    return ret;
}

int32_t read_symbol_table(const std::string& filename){
    std::string symbol_table_str = read(filename);

    std::string head;
    uint32_t offset = 0;
    const char* data = symbol_table_str.c_str();
    for(uint32_t i = 0; i < nnet_str.size(); i++){
        switch (data[i]){
            case '\t':
                head.assign(data, offset, i - offset);
                offset = i + 1;
                break;
            case '\n':
                symbol_table.insert({atoi(symbol_table_str.substr(offset, i - offset).c_str()), haad});
                offset = i + 1;
                break;
        }
    }
}

std::string read(const std::string& filename){
    std::ifstream istream(filename.c_str());
    std::stringstream ss;
    ss << istream.rdbuf();
    return ss.str();
}
