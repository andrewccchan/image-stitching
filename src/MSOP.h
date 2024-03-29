#ifndef __MSOP_H__
#define __MSOP_H__
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include "common.h"
using namespace cv;
using namespace std;

struct Corner {
    float x; // x-coor. in the down-sampled scale
    float y; // y-coor. in the down-sampled scale
    float fullX; // x-coor. in full scale
    float fullY; // y-corr. in full scale
    float r; // radius
    float f; // strength
    float o; // orientation in theta
};


class MSOP {
public:
    MSOP(int, float, float, float, int);
    void extract(Mat, vector<Descriptor>&);
private:
    void tensorConv(vector<Mat>&, float);
    void calGrad(Mat&, Mat&, int);
    void outProd(vector<Mat>&, vector<Mat>&);
    void findCandidates(vector<Mat>&, vector<Mat>&, float, vector<Corner>&);
    void adapNonMaxSup(vector<Corner>&, vector<Corner>&, bool);
    void subPixelRefine(const Mat&, Corner&);
    void calOrient(vector<Mat>&, Corner&);
    void calDescriptors(Mat&, vector<Corner>&, vector<Descriptor>&);
    void warp2Local(const Mat&, Mat&, float, Point = Point(0, 0));
    void drawCorners(const Mat&, const vector<Corner>&, int, string);
    // void writePoints(const Mat&, vector<Point>&, int, const char*);
    // void writePoints(const Mat&, vector<Corner>&, int, const char*);

private:
    int     subRate;
    float   sigP;
    float   sigD;
    float   sigI;
    int     nPts; // number of sample points
};
#endif
