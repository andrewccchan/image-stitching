#include <opencv2/opencv.hpp>
#include "common.h"
using namespace cv;

class AlignWithFeatures {
 public:
  void getGlobalShift(vector<Mat>&, vector< vector<Descriptor> >&, vector<int>&, vector<int>&);
  void getPairwiseShift(Mat&, Mat&, vector<Descriptor>&, vector<Descriptor>&, int&, int&);
};
