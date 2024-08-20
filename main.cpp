#include "layer.h"
#include <windows.h>
using namespace std;
using namespace Eigen;
MatrixXd  Y_predict;
double maxX=-100000, maxY=-100000, minX=100000, minY=100000;
// CSV ������ �о�ͼ� Eigen ��ķ� ��ȯ�ϴ� �Լ�
void readCSV(const std::string& filename, MatrixXd& X, MatrixXd& Y,MatrixXd& B) {
    ifstream file(filename);
    string line;
    vector<vector<double>> data;

    // ù ���� ����
    getline(file, line);

    // ������ �� ������ �о��
    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        vector<double> row;

        // ���� �� ������ ������ �� ���� ���� �Ľ�
        while (getline(lineStream, cell, ',')) {
            row.push_back(stod(cell)); // string�� double�� ��ȯ
        }

        data.push_back(row);
    }

    // ����� ũ�� ���� (������ ���� Y, �������� X)
    int rows = data.size();
    int cols = data[0].size() - 1;

    // X ��� �ʱ�ȭ (�� �ϳ� �߰�)
    X.resize(rows, cols + 1);
    // Y ��� �ʱ�ȭ
    Y.resize(rows, 1);
    Y_predict.resize(rows, 1);
    B.resize(cols + 1, 1);
    
    for (int i = 0; i < cols + 1; i++)
        B(i, 0) = 0;

    // �Ľ��� �����͸� X�� Y ��Ŀ� ����
    for (int i = 0; i < rows; ++i) {
        X(i, 0) = 1.0; // ù ��° ���� 1�� �߰�
        for (int j = 0; j < cols; ++j) {
            X(i, j + 1) = data[i][j]; // ���� �����͸� ù ��° �� ���ķ� �̵�
            if (X(i, j + 1) > maxX)
                maxX = X(i, j + 1);
            if (X(i, j + 1) < minX)
                minX = X(i, j + 1);

        }
        Y(i, 0) = data[i][cols];  // ������ ���� Y�� ����
        if (Y(i, 0) > maxY)
            maxY = Y(i, 0);
        if (Y(i, 0) < minY)
            minY = Y(i, 0);
    }
}

int main() {
    MatrixXd X, Y, B, temp ;
    

    // CSV ���� �б�
    readCSV("ProcessDifference_train.csv", X, Y ,B);

    temp = X.transpose() * (X);
    B = temp.inverse() * X.transpose() * Y;
    
    Y_predict = X * B;

    //  MLR 100 �� ���
    for (int i = 0; i < 100; i++) 
        cout << "Y : " << Y(i, 0) << "  Y_predict : " << Y_predict(i, 0) << endl;
    


    // MLP ����ȭ

    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            X(i, j ) = X(i, j ) / (maxX - minX);
        }
        Y(i, 0) = Y(i, 0) / (maxY - minY);
    }
    


    cout << "Test";




    // �� Layer �� �����ϴ� ����ġ���� ��ķ� ���ǵȴ�
    

    hiddenLayer hidden1(X.cols(), 32), hidden2(32, 16);
    outLayer out(16, 1);
    // ����ġ ���̾ �ʱ�ȭ
    hidden1.init();
    hidden2.init();
    out.init();


    for (int i = 0; i < 7000; i++) {

        hidden1.empty_value();
        hidden2.empty_value();
        out.empty_value();


        hidden1.feed(1, X.row(i));
        hidden2.feed(1, hidden1.Node);
        out.feed(2, hidden2.Node);


        double loss = out.calLoss(Y(i,0));

        out.back(loss, hidden2.Node);
        hidden2.back(out.diff, out.inW, hidden1.Node);
        hidden1.back(hidden2.diff, hidden2.inW, X.row(i));

        std::cout << "loss is : " << loss << endl;
        
        
        Sleep(10);


    }
    

    return 0;
}
