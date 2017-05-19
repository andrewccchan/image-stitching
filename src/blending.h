#include<vector>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void directBlending(vector<Mat>&, vector<int>&, vector<int>&);

void alphaBlending(vector<Mat>&, vector<int>&, vector<int>&);

void poissonBlending(vector<Mat>&, vector<int>&, vector<int>&);
