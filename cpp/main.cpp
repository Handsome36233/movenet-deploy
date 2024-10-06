#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>

using namespace std;
using namespace cv;

void PrintSessionInfo(const Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
    // 获取输入输出信息
    auto input_count = session.GetInputCount();
    auto output_count = session.GetOutputCount();

    cout << "Input Count: " << input_count << endl;
    cout << "Output Count: " << output_count << endl;

    // 打印输入名称和形状
    for (int i = 0; i < input_count; ++i) {
        string input_name = session.GetInputNameAllocated(i, allocator).get();
        vector<int64_t> input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

        cout << "Input " << i << " Name: " << input_name << endl;
        cout << "Input " << i << " Shape: ";
        for (const auto& dim : input_shape) cout << dim << ' ';
        cout << endl;
    }

    // 打印输出名称和形状
    for (int i = 0; i < output_count; ++i) {
        string output_name = session.GetOutputNameAllocated(i, allocator).get();
        vector<int64_t> output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

        cout << "Output " << i << " Name: " << output_name << endl;
        cout << "Output " << i << " Shape: ";
        for (const auto& dim : output_shape) cout << dim << ' ';
        cout << endl;
    }
}

void BlobFromImage(cv::Mat& iImg, uint8_t* iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = iImg.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <onnx_model_path> <image_path>\n";
        return -1;
    }
    const char* model_path = (const char *)argv[1];
    const char* image_path = (const char *)argv[2];
    const char* save_path = "output.jpg";   // 保存结果
    
    // 配置参数
    float confidenceThreshold = 0.3;
    int input_width = 256;
    int input_height = 256;

    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "movenet_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 加载模型
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    // 打印输入输出信息
    PrintSessionInfo(session, allocator);
    Ort::RunOptions options;
    vector<const char*> inputNodeNames;
    vector<const char*> outputNodeNames;

    size_t inputNodesNum = session.GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++)
    {
        Ort::AllocatedStringPtr input_node_name = session.GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session.GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++)
    {
        Ort::AllocatedStringPtr output_node_name = session.GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
    options = Ort::RunOptions{ nullptr };

    // 处理图片
    cv::Mat image = cv::imread(image_path);
    cv::Mat show_img = image.clone();
    int img_width = image.cols;
    int img_height = image.rows;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(input_width, input_height));
    // 转tensor
    uint8_t* blob = new uint8_t[image.total() * 3];
    BlobFromImage(image, blob);
    vector<int64_t> inputNodeDims = { 1, 3, input_height, input_width };
    Ort::Value inputTensor = Ort::Value::CreateTensor<uint8_t>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * input_width * input_height,
        inputNodeDims.data(), inputNodeDims.size());
    // 模型推理
    auto outputTensor = session.Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    vector<int64_t> outputNodeDims = tensor_info.GetShape();
    int num_keypoints = outputNodeDims[2];
    cout << "num_keypoints: " << num_keypoints << endl;
    // 取出输出数据
    auto output = outputTensor.front().GetTensorMutableData<float>();

    for (int i = 0; i < num_keypoints; i++) {
        for (int j = 0; j < 3; j++) {
            float score = output[i * 3 + 2];
            if (score < confidenceThreshold) continue;
            float x = output[i * 3 + 1];
            float y = output[i * 3];
            cout << "score: " << score << " x: " << x << " y: " << y << endl;
            int x1 = int(x * img_width);
            int y1 = int(y * img_height);
            cv::circle(show_img, cv::Point(x1, y1), 2, cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imwrite(save_path, show_img);
    cout << "save result to " << save_path << endl;
    return 0;
}
