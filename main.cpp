#include <stdio.h>
#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <numeric>

int main () {
	printf("LustBlock Inference Start\n");
	Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_VERBOSE, "Default"};
	Ort::SessionOptions sf;
	int device_id = 0;
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
	Ort::Session msession = Ort::Session(env, "./nsfw-detect.onnx", sf);
	Ort::AllocatorWithDefaultOptions allocator;
	size_t numInputNodes = msession.GetInputCount();
	Ort::TypeInfo inputTypeInfo = msession.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	std::vector<int64_t> indims = inputTensorInfo.GetShape();
	std::cout << "Input Dimensions: "<< inputType << "\n";
	cv::Mat img = cv::imread("./170.jpg", cv::IMREAD_COLOR);
	unsigned const img_width = img.rows;
	unsigned const img_height = img.cols;
	unsigned const img_channel = img.channels();
	std::cout << "image info: " << img_width << ", " << img_height << ", " << img_channel << std::endl;
	int size[] = { 1, img_height, img_width, img_channel };
	cv::Mat blob(4, size, CV_8U, img.data);

	std::vector<int64_t> input_node_dims(4);
	input_node_dims[0] = 1;
	input_node_dims[1] = img_height;
	input_node_dims[2] = img_width;
	input_node_dims[3] = img_channel;

	unsigned const input_tensor_size = img_width * img_height * img_channel;
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<uchar>(memory_info, blob.ptr(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	std::vector<const char*> input_node_names{ msession.GetInputName(0, allocator) };
	std::vector<const char*> output_node_names{ "num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0" };
	auto output_tensors = msession.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, input_node_names.size(), output_node_names.data(), output_node_names.size());
}
