#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace std;

void inferOnnx()
{
    std::cout << "inferOnnx begin " << std::endl;
    //设置为VERBOSE，方便控制台输出时看到是使用了cpu还是gpu执行
    //Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");

    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(10);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //CUDA option set
    OrtCUDAProviderOptions cuda_option;
    cuda_option.device_id = 0;
    cuda_option.arena_extend_strategy = 0;
    cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_option.gpu_mem_limit = SIZE_MAX;
    cuda_option.do_copy_in_default_stream = 1;

    try {
        session_options.AppendExecutionProvider_CUDA(cuda_option);
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    #ifdef _WIN32
        const wchar_t* model_path = L"./ResNet10.onnx";
    #else
        const char* model_path = "./ResNet10.onnx";
    #endif

    wprintf(L"%s\n", model_path);

    Ort::Session session(env, model_path, session_options);
    //Ort::AllocatorWithDefaultOptions allocator;
    //size_t num_input_nodes = session.GetInputCount();
    //size_t num_output_nodes = session.GetOutputCount();

    const char* input_names[] = { "input" }; // must keep the same as model export
    const char* output_names[] = { "output" };

    const int height = 128;
    const int width = 128;
    const int output_size = 30;
    const int batch_size = 1;
    const int img_channels = 2;

    std::array<float, batch_size* img_channels* height* width> input_matrix;
    std::array<float, batch_size* output_size> output_matrix;

    std::array<int64_t, 4> input_shape{ batch_size, img_channels, height, width };
    std::array<int64_t, 2> output_shape{ batch_size, output_size };

    std::vector<std::vector<std::vector<std::vector<float>>>> inputs;
    for (int i = 0; i < batch_size; i++) {
        vector<vector<vector<float>>> t1;
        for (int j = 0; j < img_channels; j++) {
            vector<vector<float>> t2;
            for (int k = 0; k < height; k++) {
                vector<float> t3;
                for (int l = 0; l < width; l++) {
                    t3.push_back((float)1.0 * (rand() % 256));
                }
                t2.push_back(t3);
            }
            t1.push_back(t2);
        }
        inputs.push_back(t1);
    }

    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < img_channels; j++)
            for (int k = 0; k < height; k++)
                for (int l = 0; l < width; l++)
                    input_matrix[i * img_channels * height * width + j * height * width + k * width + l] = inputs[i][j][k][l];

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_matrix.data(), input_matrix.size(), input_shape.data(), input_shape.size());
    try
    {
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_matrix.data(), 
            output_matrix.size(), output_shape.data(), output_shape.size());
        
        //预热
        for (int i = 0; i < 100; ++i) {
            session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
        }
        int epoch = 10000;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < epoch; ++i) {
            session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto duration_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "infer time: " << duration_milli * 1.0 / epoch << " ms" << std::endl;
        std::cout << "infer time: " << duration_nano * 1.0 / epoch << " ns" << std::endl;
        std::cout << "fps: " << epoch * 1000.0 / duration_milli << std::endl;
        std::cout << "fps: " << epoch * 1e9 * 1.0 / duration_nano << std::endl;
        std::cout << "infer done." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    vector<float> outputs;

    std::cout << "infer output: \n";
    for (int i = 0; i < output_size; i++) {
        outputs.emplace_back(output_matrix[i]);
        std::cout << outputs[i] << "\t";
    }
    std::cout << endl;

    std::cout << "inferOnnx end " << std::endl;
}

int main(int argc, char** argv)
{
    inferOnnx();
    getchar();
    return 0;
}
