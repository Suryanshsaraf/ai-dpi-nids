#include "inference_engine.h"
#include <iostream>
#include <stdexcept>

namespace DPI {

InferenceEngine::InferenceEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "AI_DPI_Inference"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{}) {
    
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Get input node name
    if (session_.GetInputCount() == 0) {
        throw std::runtime_error("ONNX model has no input nodes");
    }
    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    input_name_ = std::string(input_name_ptr.get());
    
    // Get output node name
    if (session_.GetOutputCount() == 0) {
        throw std::runtime_error("ONNX model has no output nodes");
    }
    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
    output_name_ = std::string(output_name_ptr.get());
    
    std::cout << "[InferenceEngine] Loaded ONNX model from: " << model_path << std::endl;
    std::cout << "[InferenceEngine]   Input Name:  " << input_name_ << std::endl;
    std::cout << "[InferenceEngine]   Output Name: " << output_name_ << std::endl;
}

int InferenceEngine::predict(const std::vector<float>& features) {
    if (features.size() != 5) {
        throw std::invalid_argument("Features vector size must be 5");
    }
    
    // Create input tensor shape
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(features.size())};
    
    // Create allocator and memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    
    // Wrap features vector in Ort::Value (tensor)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(features.data()),
        features.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    // Names as char pointers
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    
    // Run session
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
    
    // Get class prediction (first output tensor)
    int64_t* output_class = output_tensors[0].GetTensorMutableData<int64_t>();
    return static_cast<int>(output_class[0]);
}

} // namespace DPI
