#include "warp.h"
#include "MSOP.h"
#include "align.h"
#include "blending.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: ./stitch [setting-file-name]\n");
    return 0;
  }
  FILE *file = fopen(argv[1], "r");

  vector<Mat> imglist;
  vector<float> focal;

  const float scale = 0.8;
  char imgpath[500];
  float focal_length;
  while (fscanf(file, "%s%f", imgpath, &focal_length) == 2) {
    Mat img = imread(imgpath);
    resize(img, img, cv::Size(), scale, scale);
    imglist.push_back(img);
    focal.push_back(focal_length * scale);
  }

  MSOP msop(2, 1.0, 1.0, 1.5, 500);
  AlignWithFeatures align;

  vector<Mat> warped;
  vector< vector<Descriptor> > featlist;
  for (int i = 0; i < (int)imglist.size(); i ++) {
    Mat warpedImg;
    vector<Descriptor> feats;
    warp(imglist[i], warpedImg, focal[i]);
    warped.push_back(warpedImg);
    msop.extract(warpedImg, feats);
    featlist.push_back(feats);
  }

  vector<int> offx, offy;
  align.getGlobalShift(warped, featlist, offx, offy);
  directBlending(warped, offx, offy);

    // Load image
    // NOTE: Need to convert image to gray scale!!!

    // Mat img1 = imread("./img/temple/IMG_3955.JPG");
    // Mat img2 = imread("./img/temple/IMG_3956.JPG");
    // Mat im1, im2;
    // warp(img1, im1, 743.615);
    // warp(img2, im2, 738.955);
}
