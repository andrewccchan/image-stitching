#include <opencv2/opencv.hpp>
#include "warp.h"
using namespace cv;

Vec3f bilinearInter(Mat &img, float i, float j) {
  /*
   *     (1 - v)     v
   *  -------------------
   *  |           |     |
   *  |           |     |
   *  |     a     |  b  |  (1 - u)
   *  |           |     |
   *  |           |     |
   *  -------------------
   *  |           |     |
   *  |     c     |  d  |     u
   *  |           |     |
   *  -------------------
   */
  int bi = floor(i), bj = floor(j);
  // fill in black if out of boundary
  if (bi < 0 || bi >= img.rows - 1 || bj < 0 || bj >= img.cols - 1)
    return Vec3f(0, 0, 0);

  float u = i - floor(i);
  float v = j - floor(j);
  Vec3f a = img.at<Vec3f>(i, j);
  Vec3f b = img.at<Vec3f>(i, j + 1);
  Vec3f c = img.at<Vec3f>(i + 1, j);
  Vec3f d = img.at<Vec3f>(i + 1, j + 1);
  return (1 - v) * (1 - u) * a + v * (1 - u) * b + (1 - v) * u * c + v * u * d;
}

void warp(Mat &input, Mat &output, float focal_length) {
  Mat inputf; // float version of the input matrix
  input.convertTo(inputf, CV_32FC3);

  int output_h = input.rows; // height doesn't change after cylindrical warp
  int output_w = (int)(atan2((float)input.cols / 2, focal_length) * focal_length * 2);
  output = Mat(output_h, output_w, CV_32FC3);

  for (int i = 0; i < output.rows; i ++) {
    for (int j = 0; j < output.cols; j ++) {
      float theta = (j - (float)output.cols / 2) / focal_length;
      float height = (i - (float)output.rows / 2) / focal_length;

      float pos_x = tan(theta) * focal_length + (float)inputf.cols / 2;
      float pos_y = height * hypot(pos_x - (float)inputf.cols / 2, focal_length) + (float)inputf.rows / 2;

      output.at<Vec3f>(i, j) = bilinearInter(inputf, pos_y, pos_x);
    }
  }
  output.convertTo(output, CV_8UC3);
}
