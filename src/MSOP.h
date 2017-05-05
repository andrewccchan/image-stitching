#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
using namespace cv;
using namespace std;

struct Corner {
    float x;
    float y;
    float r; // radius
    float f; // strength
    float o; // orientation in theta
};

struct Descriptor {
    int x; // x-coor. in  global coordinate
    int y; // y-coor. in  gloabl coordinate
    float fea[64];
};

class MSOP {
public:
    MSOP(int, float, float, float, int);
    void extract(Mat);
private:
    void tensorConv(vector<Mat>&, float);
    void calGrad(Mat&, Mat&, int);
    void outProd(vector<Mat>&, vector<Mat>&);
    void findCandidates(vector<Mat>&, Mat&, vector<Point>&);
    void adapNonMaxSup(const Mat&, const vector<Point>&, vector<Corner>&);
    void subPixelRefine(const Mat&, vector<Corner>&);
    void calOrient(vector<Mat>&, vector<Corner>&);
    void calDescriptors(Mat&, int, vector<Corner>&, vector<Descriptor>&);
    void warp2Local(const Mat&, Mat&, float, Point = Point(0, 0));
    void drawCorners(const Mat&, int, const vector<Corner>&, const char*);
    // void writePoints(const Mat&, vector<Point>&, int, const char*);
    // void writePoints(const Mat&, vector<Corner>&, int, const char*);

private:
    int     subRate;
    float   sigP;
    float   sigD;
    float   sigI;
    int     nPts; // number of sample points
};