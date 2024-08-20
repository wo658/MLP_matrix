#include "layer.h"
#include <cmath>  // std::exp 사용을 위해 포함
#include <random>

double learning_rate = 0.001;
// Sigmoid 함수 구현
double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}
RowVectorXd sigmoid_v(RowVectorXd x) {

	for(int i=0;i<x.cols();i++)
		x(0,i) = 1.0 / (1.0 + std::exp(-x(0,i)));
	return x;
}


// Sigmoid 함수의 미분 계산
double sigmoid_derivative(double x) {
	double sig = sigmoid(x);
	return sig * (1.0 - sig);
}
// version 1 = Sigmoid , version 2 = Linear
void Layer::feed(int version, RowVectorXd preNode) {
	//std::cout<< " " << (preNode * this->inW).cols() << " ";
	this->noActivationNode = preNode * this->inW + this->bias;
	if (version == 1)
		this->Node = sigmoid_v(noActivationNode);
	else
		this->Node = noActivationNode;
}
Layer::Layer(int pre_num, int num ) : pre_num(pre_num), num(num){


	this->inW.resize(pre_num, num);
	this->Node.resize(1, num);
	this->bias.resize(1, num);
	this->diff.resize(1, num);
	this->noActivationNode(1, num);

};
hiddenLayer::hiddenLayer(int pre_num, int num) : Layer(pre_num, num) {};
outLayer::outLayer(int pre_num, int num) : Layer(pre_num, num) {};
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> d(0, 1);

void Layer::init() {
	for (int j = 0; j < this->inW.cols(); ++j) {
		for (int i = 0;i < this->inW.rows(); ++i) {
			this->inW(i, j) = d(gen);
		}
		this->bias(0, j) = d(gen);


	}
}

void Layer::empty_value() {

	for (int i = 0; i < this->Node.rows(); ++i) {
		for (int j = 0; j < this->Node.cols(); ++j) {
			this->Node(i, j) = 0;
		}


	}
}

void outLayer::back(double loss, RowVectorXd preLayerActivae) {
	RowVectorXd temp;
	temp.resize(1, this->num);


	temp = 2 * loss * noActivationNode;
	this->diff = temp;
	double dw = temp * preLayerActivae.transpose();
	for (int i = 0; i < inW.cols(); i++) {
		for (int j = 0; j < inW.rows(); j++)
			this->inW(j, i) = this->inW(j, i) - learning_rate * dw;
		this->bias(0,0) = this->bias(0,0) - learning_rate * dw;
	}
}
double outLayer::calLoss(double Y) {
	std::cout << this->Node(0,0);
	return (Y - this->Node(0, 0) * (Y - this->Node(0, 0)));
}


void hiddenLayer::back(RowVectorXd diff, MatrixXd nextLayerW, RowVectorXd preLayerActivate) {
	RowVectorXd temp;
	temp.resize(1, this->num);
	temp = (nextLayerW * diff.transpose()).transpose();
	for (int i = 0; i < temp.cols(); i++)
		temp = temp * sigmoid_derivative(noActivationNode(0, i));
	diff = temp;
	double dw = temp * preLayerActivate.transpose();

	for (int i = 0; i < inW.cols(); i++) {
		for (int j = 0; j < inW.rows(); j++)
			this->inW(i, j) = this->inW(i, j) - learning_rate * dw;
		this->bias(0, i) = this->bias(0, i) - learning_rate * dw;
	}
}