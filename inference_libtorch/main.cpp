#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <ctime>
#include <ratio>
#include <chrono>


#include <fstream>
#include <string>
#include <stdio.h>
#include <cmath>
#include <sstream>

using namespace std::chrono;
using namespace std;

double sigmoid(double input)
{
  return 1/(1+exp(-input));
}

int main() {
  torch::jit::script::Module module;
  try {
    module = torch::jit::load("/home/kim/mocca/inference/model_100_10.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

int seq_num = 50;
int input_feat_num = 6;

torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false).device(torch::kCPU);
torch::Tensor input_tensor = torch::ones({1, seq_num, input_feat_num}, options);
// for (int seq = 0; seq < seq_num; seq++)
//   for (int input_feat = 0; input_feat < input_feat_num; input_feat++)
//     input_tensor[0][seq][input_feat] = 1.0;

// torch::Tensor input_tensor = torch::ones({1, 3, 224, 224});

std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);

// Execute the model and turn its output into a tensor.
double resi_max[2] = {76.35, 25.274};
double resi_min[2] = {-76.35, -25.274};
double threshold[2] = {5.9298, 2.5928};

at::Tensor output = module.forward(inputs).toTensor();
for (int i=0; i<10; i++){
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  
  output = module.forward(inputs).toTensor();

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double, std::milli> time_span = t2 - t1;
  std::cout << "It took me " << time_span.count() << " milliseconds."<<std::endl;
}
double lstm_out_1 = (resi_max[0] - resi_min[0]) * output[0][0].item<double>()/2.0 + (resi_max[0]+resi_min[0])/2.0;
double lstm_out_2 = (resi_max[1] - resi_min[1]) * output[0][1].item<double>()/2.0 + (resi_max[1]+resi_min[1])/2.0;

std::cout << "LSTM Joint 1: " << lstm_out_1 << '\n';
std::cout << "LSTM Joint 2: " << lstm_out_2 << '\n';

bool collision_detection = false;

}
