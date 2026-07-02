#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace DPI {

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine() = default;
    
    // Returns 1 if anomalous/malicious, 0 if normal.
    // Throws exception if prediction fails.
    int predict(const std::vector<float>& features);

private:
    Ort::Env env_;
    Ort::Session session_;
    
    std::string input_name_;
    std::string output_name_;
};

} // namespace DPI

#endif // INFERENCE_ENGINE_H
