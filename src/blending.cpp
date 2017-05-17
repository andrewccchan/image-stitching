#include "blending.h"
using namespace cv;

void directBlending(vector<Mat> &img, vector<int> &offx, vector<int> &offy) {
  int width = 0, height = 0;
  for (int i = 0; i < (int)img.size(); i ++) {
    width = max(width, offx[i] + img[i].cols);
    height = max(height, offy[i] + img[i].rows);
  }

  Mat combined = Mat::zeros(height, width, CV_8UC3);

  for (int k = 0; k < (int)img.size(); k ++) {
    for(int i = 0; i < img[k].rows; i++)
      for(int j = 0; j < img[k].cols; j++)
        combined.at<Vec3b>(i + offy[k], j + offx[k]) = img[k].at<Vec3b>(i, j);
  }

  imwrite("combined.jpg", combined);
}

