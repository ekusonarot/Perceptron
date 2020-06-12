
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <ctime>

#include "weight.h"
#include "linear.cuh"

int main() {
	Linear linear("input", 2, 2);
	Linear linear1("output", 2, 1, &linear);
	float input0[2] = { 0,0 }, input1[2] = { 0,1 }, input2[2] = { 1,0 }, input3[2] = { 1,1 };
	float target0[1] = { 1 }, target1[1] = { 0 }, target2[1] = { 0 }, target3[1] = { 1 };
	for (int i = 0; i < 100000; i++) {
		linear.setInput(input0);
		linear.forward();
		linear1.forward();
		linear1.backward(target0);
		linear.backward();
		if (i % 1001 == 0)
			std::cout << input0[0] << " " << input0[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;

		linear.setInput(input1);
		linear.forward();
		linear1.forward();
		linear1.backward(target1);
		linear.backward();
		if (i % 1001 == 0)
			std::cout << input1[0] << " " << input1[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;

		linear.setInput(input2);
		linear.forward();
		linear1.forward();
		linear1.backward(target2);
		linear.backward();
		if (i % 1001 == 0)
			std::cout << input2[0] << " " << input2[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;

		linear.setInput(input3);
		linear.forward();
		linear1.forward();
		linear1.backward(target3);
		linear.backward();
		if (i % 1001 == 0)
			std::cout << input3[0] << " " << input3[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;
	}


	linear.setInput(input0);
	linear.forward();
	linear1.forward();
	std::cout << input0[0] << " " << input0[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;

	linear.setInput(input1);
	linear.forward();
	linear1.forward();
	std::cout << input1[0] << " " << input1[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;

	linear.setInput(input2);
	linear.forward();
	linear1.forward();
	std::cout << input2[0] << " " << input2[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;

	linear.setInput(input3);
	linear.forward();
	linear1.forward();
	std::cout << input3[0] << " " << input3[1] << " " << "=" << " " << linear1.getOutput()[0] << std::endl;
	
	linear.saveWeight();
	linear1.saveWeight();
}