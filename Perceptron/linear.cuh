
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "layer.cuh"
#include "weight.h"


__global__
void eval(float*, float*, float*, float*, int, int);
__global__
void back(float*, float*, float*, float*, float*, float*, int, int, int);
__global__
void back(float*, float*, float*, float*, float*, float*, int, int);
__device__
float ETA = 0.01;
__device__
float activate(float);
__device__
float deriv_activate(float);

class Linear : public Layer {
public:
	Linear(char* name, int input_length, int output_length, Layer* parent) {
		this->name = name;
		this->input_length = input_length;
		this->output_length = output_length;
		this->parent = parent;
		parent->setChild(this);

		
		weight = new float[(input_length + 1) * output_length];
		output = new float[output_length];


		this->d_input = parent->d_output_activated;
		cudaMalloc(&d_weight, (input_length + 1) * output_length * sizeof(float));
		cudaMalloc(&d_output, output_length * sizeof(float));
		cudaMalloc(&d_output_activated, output_length * sizeof(float));
		cudaMalloc(&d_target, output_length * sizeof(float));
		cudaMalloc(&d_delta, output_length * sizeof(float));

		getweight(weight, (input_length + 1) * output_length, name);
		cudaMemcpy(d_weight, weight, (input_length + 1) * output_length * sizeof(float), cudaMemcpyHostToDevice);
	}
	Linear(char* name, int input_length, int output_length) {
		this->name = name;
		this->input_length = input_length;
		this->output_length = output_length;

		
		weight = new float[(input_length + 1) * output_length];
		output = new float[output_length];


		cudaMalloc(&d_input, input_length * sizeof(float));
		cudaMalloc(&d_weight, (input_length + 1) * output_length * sizeof(float));
		cudaMalloc(&d_output, output_length * sizeof(float));
		cudaMalloc(&d_output_activated, output_length * sizeof(float));
		cudaMalloc(&d_target, output_length * sizeof(float));
		cudaMalloc(&d_delta, output_length * sizeof(float));

		getweight(weight, (input_length + 1) * output_length, name);
		cudaMemcpy(d_weight, weight, (input_length + 1) * output_length * sizeof(float), cudaMemcpyHostToDevice);
	}
	~Linear() {
		if (d_input != nullptr)
			cudaFree(d_input);
		cudaFree(d_weight);
		cudaFree(d_output);
		cudaFree(d_output_activated);
		cudaFree(d_delta);
		cudaFree(d_target);
		
		delete(weight);
		delete(output);
	}
	virtual void setChild(Layer* child) {
		this->child = child;
	}
	virtual void forward();
	virtual void backward();
	virtual void backward(float*);
	virtual void setInput(float*);
	virtual float* getOutput();
	virtual void saveWeight();
};

void Linear::saveWeight()
{
	cudaMemcpy(weight, d_weight, (input_length + 1) * output_length * sizeof(float), cudaMemcpyDeviceToHost);
	saveweight(weight, (input_length + 1) * output_length, this->name);
}

void Linear::forward()
{
	eval<<<1,output_length>>>(d_input, d_weight, d_output, d_output_activated, input_length, output_length);
}
void Linear::backward()
{
	dim3 grid(1, 1, 1);
	dim3 block(input_length + 1, output_length, 1);
	back << <grid, block >> > (d_input, d_weight, d_output, d_delta, child->d_delta, child->d_weight, input_length, output_length, child->output_length);
}
void Linear::backward(float* target)
{
	cudaMemcpy(d_target, target, output_length * sizeof(float), cudaMemcpyHostToDevice);
	dim3 grid(1, 1, 1);
	dim3 block(input_length + 1, output_length, 1);
	back << <grid, block >> > (d_input, d_weight, d_output, d_output_activated, d_target, d_delta, input_length, output_length);
}

void Linear::setInput(float* input)
{
	cudaMemcpy(d_input, input, input_length * sizeof(float), cudaMemcpyHostToDevice);
}
float* Linear::getOutput()
{
	cudaMemcpy(output, d_output_activated, output_length * sizeof(float), cudaMemcpyDeviceToHost);
	return output;
}

__global__
void eval(float* input, float* weight, float* output, float* output_activated, int input_length, int output_length)
{
	const int xidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (xidx < output_length) {
		int i = 0;
		output[xidx] = 0;
		for (i = 0; i < input_length; i++) {
			output[xidx] += input[i] * weight[output_length * i + xidx];
		}
		output[xidx] += weight[output_length * i + xidx];
	}
	output_activated[xidx] = activate(output[xidx]);
}
__global__
void back(float* input, float* weight, float* output, float* delta, float* n_delta, float* n_weight, int input_length, int output_length, int n_output_length)
{
	const int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int yidx = blockIdx.y * blockDim.y + threadIdx.y;

	if (xidx < 1 && yidx < output_length) {
		delta[yidx] = 0;
		for (int i = 0; i < n_output_length; i++)
			delta[yidx] += n_delta[i] * n_weight[n_output_length * yidx + i];
		delta[yidx] *= deriv_activate(output[yidx]);
	}
	__syncthreads();
	if (xidx < input_length && yidx < output_length) {
		weight[output_length * xidx + yidx] += ETA * delta[yidx] * input[xidx];
	}
	if (xidx == input_length && yidx < output_length)
		weight[output_length * xidx + yidx] += ETA * delta[yidx];
}
__global__
void back(float* input, float* weight, float* output, float* output_activated, float* target, float* delta, int input_length, int output_length)
{
	const int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int yidx = blockIdx.y * blockDim.y + threadIdx.y;

	if (xidx < 1 && yidx < output_length) {
		delta[yidx] = deriv_activate(output[yidx]) * (target[yidx] - output_activated[yidx]);
	}
	
	__syncthreads();
	if (xidx < input_length && yidx < output_length) {
		weight[output_length * xidx + yidx] += ETA * delta[yidx] * input[xidx];
	}
	if (xidx == input_length && yidx < output_length)
		weight[output_length * xidx + yidx] += ETA * delta[yidx];
	
}
/* sigmoid 
	ETA : about 0.4
*/
/*
__device__
float activate(float x)
{
	return 1.0 / (1.0 + exp(-x));
}
__device__
float deriv_activate(float x)
{
	return activate(x)*(1-activate(x));
}
*/
/* leaky relu 
	ETA : about 0.01
*/
__device__
float activate(float x)
{
	if (0 <= x) return x;
	return 0.01 * x;
}
__device__
float deriv_activate(float x)
{
	if (0 <= x) return 1;
	return 0.01;
}
#endif // LINEAR_CUH