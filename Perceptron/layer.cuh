#ifndef LAYER_CUH
#define LAYER_CUH


class Layer {
public:
	Layer* parent=nullptr;
	Layer* child=nullptr;
	char* name=nullptr;
	int input_length, output_length;
	float* weight, * output;
	float* d_input = nullptr, * d_weight = nullptr, * d_output = nullptr, * d_delta = nullptr, * d_output_activated=nullptr, * d_target=nullptr;
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void backward(float*) = 0;
	virtual void setChild(Layer*) = 0;
	virtual void setInput(float*) = 0;
	virtual float* getOutput() = 0;
	virtual void saveWeight() = 0;
};

#endif // LAYER_CUH