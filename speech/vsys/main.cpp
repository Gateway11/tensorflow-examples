//
//  main.cpp
//  test_tf
//
//  Created by 薯条 on 2018/8/17.
//  Copyright © 2018年 薯条. All rights reserved.
//

#include <time.h>
#include <iostream>
#include <tensorflow/c/c_api.h>

constexpr char* nnet_path = "output_graph.pb";

int main(int argc, const char * argv[]) {
    std::unique_ptr<FILE, void (*)(FILE *)> f(fopen(nnet_path, "rb"), [](FILE* f){if(f) fclose(f);});
    uint32_t size = 0;
    uint8_t* data = nullptr;
    if(f){
        fseek(f.get(), 0, SEEK_END);
        size = ftell(f.get());
        data = new uint8_t[size];
        fseek(f.get(), 0, SEEK_SET);
        size_t read = fread(data, 1, size, f.get());
    }else{
        return -1;
    }
    
    //创建图
    TF_Graph* graph = TF_NewGraph();
    
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_Buffer* buf = TF_NewBufferFromString(data, size);
    TF_Status* status = TF_NewStatus();
    delete [] data;
    
    TF_GraphImportGraphDef(graph, buf, opts, status);
    TF_Code tf_code = TF_GetCode(status);
    if(tf_code != TF_OK){
        printf("TF_GraphImportGraphDef failed: %d\n", tf_code);
    }
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buf);
    
    //创建session
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Session* sess = TF_NewSession(graph, session_opts, status);
    if(tf_code != TF_OK){
        printf("TF_NewSession failed: %d\n", tf_code);
    }
    TF_DeleteSessionOptions(session_opts);
    
    const int ninputs = 2;
    const int noutputs = 1;
    const int ntargets = 1;
    
    TF_Output inputs[ninputs];
    TF_Tensor* input_values[ninputs];
    
    //input arg1
    TF_Operation* oper_input = TF_GraphOperationByName(graph, "input");
    inputs[0] = TF_Output{oper_input, 0};
    int64_t input_dims[] = {1, 88, 494};
    input_values[0] = TF_AllocateTensor(TF_FLOAT, input_dims, sizeof(input_dims),
                                        input_dims[0] * input_dims[1] * input_dims[2] * sizeof(float));
    
    //input arg2
    TF_Operation* oper_seq_length = TF_GraphOperationByName(graph, "seq_length");
    inputs[1] = TF_Output{oper_seq_length, 0};
    int64_t input_seq_len[] = {1};
    input_values[1] = TF_AllocateTensor(TF_FLOAT, input_dims, sizeof(input_seq_len), input_seq_len[0] * sizeof(int32_t));
    
    //output
    TF_Operation* oper_output = TF_GraphOperationByName(graph, "decode/CTCBeamSearchDecoder");
    TF_Output outputs = {oper_output, noutputs};
    int64_t word_size = 1788;
    int64_t output_dims[] = {1, word_size};
    TF_Tensor* tensor_output = TF_AllocateTensor(TF_FLOAT, output_dims, 2, word_size * sizeof(float));
    
    float* tensor_input_data = (float *)TF_TensorData(input_values[0]);
    int32_t* tensor_input_seq_len = (int32_t *)TF_TensorData(input_values[1]);
    *tensor_input_seq_len = 88;
    
    srand (time(NULL));
    for (int i=0; i < (*tensor_input_seq_len) * input_dims[2]; i++) {
        tensor_input_data[i] = (rand() / (float)RAND_MAX) * 2 - 1;
    }
    //TODO
    
    TF_SessionRun(sess, nullptr, inputs, input_values, ninputs,
                  &outputs, &tensor_output, noutputs, &oper_output, ntargets, nullptr, status);
    
    TF_CloseSession(sess, status);
    TF_DeleteSession(sess, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return 0;
}
