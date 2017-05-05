#include "MSOP.h"

int main() {
    MSOP msop(2, 1.0, 1.0, 1.5, 500);
    // Load image
    // NOTE: Need to convert image to gray scale!!!
    Mat img = imread("./img/horse.png");
    msop.extract(img);
}
