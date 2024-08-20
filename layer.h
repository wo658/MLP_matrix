#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;


// Input , Output , Hidden
class Layer {

public:
	// 해당 Layer 에 들어오는 W 행렬
	MatrixXd  inW;
	int pre_num, num;
	Layer(int pre_num, int num );
	RowVectorXd Node, diff, bias , noActivationNode;
	void feed(int version , RowVectorXd preNode);
	void init();
	void empty_value();


};
class hiddenLayer : public Layer {
public:
	hiddenLayer(int pre_num, int num);
	void back(RowVectorXd diff, MatrixXd nextLayerW, RowVectorXd preLayerActivate);
};
class outLayer : public Layer {
public:
	outLayer(int pre_num, int num);
	double calLoss(double Y , int i);
	void back(double loss,RowVectorXd preLayerActivae);

};







