#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include <unistd.h>
#include <math.h> 

//#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This file is based on the TF lite minimal example where the
// "BuiltinOpResolver" is modified to include the "Larq Compute Engine" custom
// ops. Here we read a binary model from disk and perform inference by using the
// C++ interface. 

// using namespace cv;
using namespace std;
using namespace tflite;


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }


void ProcessInputWithFloatModel(uint8_t* input, float* buffer, int H, int W, int C) {
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (C == 3) {
        *(buffer + (y * W * C) + (x * C) + 0) = *(input + (y * W * C) + (x * C) + 2) / 255.0f;
        *(buffer + (y * W * C) + (x * C) + 1) = *(input + (y * W * C) + (x * C) + 1) / 255.0f;
        *(buffer + (y * W * C) + (x * C) + 2) = *(input + (y * W * C) + (x * C) + 0) / 255.0f;
      } else {
        for (int c = 0; c < C; ++c) {
          *(buffer + (y * W * C) + (x * C) + 0) / 255.0f;
        }
      }
    }
  }
}

void dummyInput(float* buffer, int H, int W, int C) {
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (C == 3) {
        *(buffer + (y * W * C) + (x * C) + 0) = 0.0;
        *(buffer + (y * W * C) + (x * C) + 1) = 0.0;
        *(buffer + (y * W * C) + (x * C) + 2) = 0.0;
      } else {
        for (int c = 0; c < C; ++c) {
          *(buffer + (y * W * C) + (x * C) + 0) / 255.0f;
        }
      }
    }
  }
}

std::string flatten2string(float* input, int nFeature) {
  std::string floatString;
  for (int i = 0; i < nFeature; ++i) {
    floatString.append(std::to_string(*(input + i)));
    floatString.append(";");
  }
  floatString.pop_back();
  return floatString;
}

std::vector<std::string> featuresByChannel(float* input, int H, int W, int nChannel) {
  std::vector<std::string> strings;
  int nString = (H * W);
  for (int f = 0; f < nString; ++f) {
    std::string floatString;
    for (int i = 0; i < nChannel; ++i) {
      floatString.append(std::to_string(*(input + i + f*nChannel)));
      floatString.append(";");
    }
    floatString.pop_back();
    strings.push_back(floatString);
  }
  return strings;
}

void print_vector_to_stdout(std::vector<std::string> v) {
  for (std::string str : v) {
    std::cout << str << std::endl;
  }
}



int main(int argc, char* argv[]) {
/*
  if (argc != 3) {
    fprintf(stderr, "lce_test <tflite model> <input_image>\n");
    return 1;
  }
*/
  const char* filename;
  const char* input_fn;
  const char* output_fn;
  bool write_to_file = false;
  bool bench_mode = false;
  int num_runs = 50;
  bool verbose = false;
  bool print_to_stdout = false;
  int num_threads = 4;
  
  for(;;) {
    switch(getopt(argc, argv, "g:i:o:bn:t:vph")) {
    
      case 'g':
        filename = optarg;
        continue;
        
      case 'i':
        input_fn = optarg;
        continue;
        
      case 'o':
        output_fn = optarg;
        write_to_file = true;
        continue;
        
      case 'b':
        bench_mode = true;
        continue;
      
      case 'n':
        sscanf(optarg, "%d", &num_runs); 
        continue;
        
      case 't':
        sscanf(optarg, "%d", &num_threads); 
        continue;
        
      case 'v':
        verbose = true;
        continue;
        
      case 'p':
        print_to_stdout = true;
        continue;
        
      case 'h':
        std::cout << "MANDATORY parameter: " << std::endl;
        std::cout << "\t-g: model file " << std::endl;
        std::cout << "BASIC USE: FEATURE EXTRACTOR" << std::endl;
        std::cout << "\t-i: input image file " << std::endl;
        std::cout << "\t-o: output feature file (OPTIONAL)" << std::endl;
        std::cout << "\t-v: verbose mode." << std::endl;
        std::cout << "\t-p: print features to stdout." << std::endl;
        std::cout << "\t-t: number of threads. Default 4. " << std::endl;
        std::cout << std::endl;
        std::cout << "BENCHMARK MODE: " << std::endl;
        std::cout << "\t-b: to enable the bechmark mode. " << std::endl;
        std::cout << "\t-n: number of runs to average. Default 50. " << std::endl;
        std::cout << "\t-t: number of threads. Default 4. " << std::endl;
        
        continue;
        
      case -1:
        break;
    }
    break;
  }
  

  int msg = 0;

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  

  // Determine the input shape 
  const std::vector<int>& t_inputs = interpreter->inputs();
  TfLiteTensor* itensor = interpreter->tensor(t_inputs[0]);

  int H = itensor->dims->data[1];
  int W = itensor->dims->data[2];
  int C = itensor->dims->data[3];
  
  if (verbose) {
    std::cout << ++msg << ") " << "Model input shape: (" << H << "," << W << "," << C << ")" << std::endl;
    std::cout << ++msg << ") " << "Input image will be resized accordingly." << std::endl;
  }

  /*
  cv::Mat raw_cvimg = cv::imread(input_fn);
  if(raw_cvimg.data == NULL) {
    printf("=== IMAGE READ ERROR ===\n");
    return 0;
  }

  if (verbose) {
    std::cout << ++msg << ") " << "image 1 loaded: " << input_fn << std::endl;
    std::cout << "\tWidth : " << raw_cvimg.cols << std::endl;
    std::cout << "\tHeight : " << raw_cvimg.rows << std::endl << std::endl;  
  }

  cv::Mat cvimg;
  cv::resize(raw_cvimg, cvimg, cv::Size(H,W));

  if (verbose) {
    std::cout << ++msg << ") " << "image 1 resized: " << input_fn << std::endl;
    std::cout << "\tWidth : " << cvimg.cols << std::endl;
    std::cout << "\tHeight : " << cvimg.rows << std::endl; 
  }
  
  */

  int nInput = interpreter->inputs()[0];
  float* input = interpreter->typed_input_tensor<float>(nInput);

  /*
  uint8_t* in = cvimg.ptr<uint8_t>(0);
  ProcessInputWithFloatModel(in, input, H, W, C);
  */
  
  interpreter->SetNumThreads(num_threads);
  
  if (bench_mode == true) {

    // Dummy Input, a real image is not needed for the benchmarch
    dummyInput(input, H, W, C);
    std::cout << "BENCHMARK:" << std::endl; 
    std::cout << "\tWarmup..." << std::endl; 
    // warmup
    for (int i = 0; i < 10; ++i) {
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    }
    std::vector<float> measures = {};  
    float total_duration = 0.0;
    for (int i = 0; i < num_runs; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
      auto stop = std::chrono::high_resolution_clock::now();
  
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>( stop - start ).count();
      measures.push_back(duration);
      total_duration += duration;
    }
    float avg = total_duration / num_runs;
    float std = 0.0;
    for (std::vector<float>::iterator it = measures.begin() ; it != measures.end(); ++it) {
      std += pow((*it - avg),2);
    }
    std = std / num_runs;
    std = sqrt(std);
    
    std::cout << "Benchmark result:" << std::endl;
    std::cout << "\tNum Runs: " << num_runs << std::endl;
    std::cout << "\tThreads: " << num_threads << std::endl;
    std::cout << "\taverage: " << avg << std::endl;
    std::cout << "\tstd dev: " << std << std::endl; 
      
  } else {
  
    cv::Mat raw_cvimg = cv::imread(input_fn);
    if(raw_cvimg.data == NULL) {
      printf("=== IMAGE READ ERROR ===\n");
      return 0;
    }

    if (verbose) {
      std::cout << ++msg << ") " << "image 1 loaded: " << input_fn << std::endl;
      std::cout << "\tWidth : " << raw_cvimg.cols << std::endl;
      std::cout << "\tHeight : " << raw_cvimg.rows << std::endl;  
    }

    cv::Mat cvimg;
    cv::resize(raw_cvimg, cvimg, cv::Size(H,W));

    if (verbose) {
      std::cout << ++msg << ") " << "image resized to fit the model input shape: " << std::endl;
      std::cout << "\tWidth : " << cvimg.cols << std::endl;
      std::cout << "\tHeight : " << cvimg.rows << std::endl; 
    }
  
    // Load the image into the model buffer
    uint8_t* in = cvimg.ptr<uint8_t>(0);
    ProcessInputWithFloatModel(in, input, H, W, C);
    // No bench, than just invoke once and write the features
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    //tflite::PrintInterpreterState(interpreter.get());

    // Read output buffers
    // get the output shape
    const std::vector<int>& t_outputs = interpreter->outputs();
    TfLiteTensor* oTensor = interpreter->tensor(t_outputs[0]);
    int features_shape = oTensor->dims->size;
    int B = oTensor->dims->data[0];
    int Hf = oTensor->dims->data[1];
    int Wf;
    int Cf;
    int feature_size;
  
    if (features_shape == 2) {
      Wf = 0;
      Cf = 0;
      feature_size = Hf;
      if (verbose)
      std::cout << ++msg << ") " << "Model out shape: (" << Hf << ", )" << std::endl;
    } else {
      Wf = oTensor->dims->data[2];
      Cf = oTensor->dims->data[3];    
      feature_size = Hf * Wf * Cf;
      if (verbose)
        std::cout << ++msg << ") " << "Model out shape: (" << Hf << "," << Wf << "," << Cf << ")" << std::endl;
    }
  
  
 
    float* output = interpreter->typed_output_tensor<float>(0);
  
    std::string str = flatten2string(output, feature_size);
    
    if (write_to_file) {
      std::ofstream outf(output_fn);
      outf << str;
      outf.close();
    }
    
    if (print_to_stdout)
      std::cout << str << std::endl;
  
    return 0;
  }
}
