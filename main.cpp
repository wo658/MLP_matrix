#include "layer.h"
#include <windows.h>
using namespace std;
using namespace Eigen;
MatrixXd  Y_predict;
double maxX=-100000, maxY=-100000, minX=100000, minY=100000;
// CSV 파일을 읽어와서 Eigen 행렬로 변환하는 함수
void readCSV(const std::string& filename, MatrixXd& X, MatrixXd& Y,MatrixXd& B) {
    ifstream file(filename);
    string line;
    vector<vector<double>> data;

    // 첫 줄을 무시
    getline(file, line);

    // 파일을 행 단위로 읽어옴
    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        vector<double> row;

        // 행을 열 단위로 나누어 각 셀의 값을 파싱
        while (getline(lineStream, cell, ',')) {
            row.push_back(stod(cell)); // string을 double로 변환
        }

        data.push_back(row);
    }

    // 행렬의 크기 설정 (마지막 열은 Y, 나머지는 X)
    int rows = data.size();
    int cols = data[0].size() - 1;

    // X 행렬 초기화 (열 하나 추가)
    X.resize(rows, cols + 1);
    // Y 행렬 초기화
    Y.resize(rows, 1);
    Y_predict.resize(rows, 1);
    B.resize(cols + 1, 1);
    
    for (int i = 0; i < cols + 1; i++)
        B(i, 0) = 0;

    // 파싱한 데이터를 X와 Y 행렬에 삽입
    for (int i = 0; i < rows; ++i) {
        X(i, 0) = 1.0; // 첫 번째 열에 1을 추가
        for (int j = 0; j < cols; ++j) {
            X(i, j + 1) = data[i][j]; // 기존 데이터를 첫 번째 열 이후로 이동
            if (X(i, j + 1) > maxX)
                maxX = X(i, j + 1);
            if (X(i, j + 1) < minX)
                minX = X(i, j + 1);

        }
        Y(i, 0) = data[i][cols];  // 마지막 열은 Y로 저장
        if (Y(i, 0) > maxY)
            maxY = Y(i, 0);
        if (Y(i, 0) < minY)
            minY = Y(i, 0);
    }
}

int main() {
    MatrixXd X, Y, B, temp ;
    

    // CSV 파일 읽기
    readCSV("ProcessDifference_train.csv", X, Y ,B);

    temp = X.transpose() * (X);
    B = temp.inverse() * X.transpose() * Y;
    
    Y_predict = X * B;

    //  MLR 100 개 출력
    for (int i = 0; i < 100; i++) 
        cout << "Y : " << Y(i, 0) << "  Y_predict : " << Y_predict(i, 0) << endl;
    


    // MLP 정규화

    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            X(i, j ) = X(i, j ) / (maxX - minX);
        }
        Y(i, 0) = Y(i, 0) / (maxY - minY);
    }
    


    cout << "Test";




    // 각 Layer 를 연결하는 가중치들이 행렬로 정의된다
    

    hiddenLayer hidden1(X.cols(), 32), hidden2(32, 16);
    outLayer out(16, 1);
    // 가중치 바이어스 초기화
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
