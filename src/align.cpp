#include <algorithm>
#include "align.h"
using namespace cv;
using namespace std;

void AlignWithFeatures::getGlobalShift(vector<Mat> &img, vector< vector<Descriptor> > &feats, vector<int> &offx, vector<int> &offy) {
  offx.clear();
  offy.clear();
  offx.push_back(0);
  offy.push_back(0);
  for (int i = 0; i + 1 < (int)img.size(); i ++) {
    int sx, sy;
    getPairwiseShift(img[i], img[i + 1], feats[i], feats[i + 1], sx, sy);
    sx += offx.back();
    sy += offy.back();
    offx.push_back(sx);
    offy.push_back(sy);
  }
  // handle global y-directional drift
  int global_y_offset = offy.back();
  for (int i = 0; i < (int)offy.size(); i ++) {
    offy[i] -= global_y_offset * i / ((int)offy.size() - 1);
  }

  int minx = *min_element(offx.begin(), offx.end());
  int miny = *min_element(offy.begin(), offy.end());
  transform(offx.begin(), offx.end(), offx.begin(), [&](int x) { return x - minx; });
  transform(offy.begin(), offy.end(), offy.begin(), [&](int y) { return y - miny; });
}

void AlignWithFeatures::getPairwiseShift(Mat &img1, Mat &img2, vector<Descriptor> &feat1, vector<Descriptor> &feat2, int &shift_x, int &shift_y) {
  Mat img1_match = img1.clone();
  Mat img2_match = img2.clone();

  printf( "feature sizes %d --- %d\n" , (int)feat1.size() , (int)feat2.size() );
  Mat feat1_mat((int)feat1.size(), 64, CV_32F);
  Mat feat2_mat((int)feat2.size(), 64, CV_32F);
  for (size_t i = 0; i < feat1.size(); i ++)
    for (int j = 0; j < 64; j ++)
      feat1_mat.at<float>(i, j) = feat1[i].fea[j];
  for (size_t i = 0; i < feat2.size(); i ++)
    for (int j = 0; j < 64; j ++)
      feat2_mat.at<float>(i, j) = feat2[i].fea[j];

  flann::KDTreeIndexParams fnn_param(2);
  flann::Index fnn1(feat1_mat, fnn_param);
  flann::Index fnn2(feat2_mat, fnn_param);
  vector<MatchPts> match_pts;

  Mat compare = Mat::zeros(img1.rows, img1.cols +img2.cols, CV_8UC3);

  for(int i = 0; i < img1.rows; i++)
    for(int j = 0; j < img1.cols; j++)
      compare.at<Vec3b>(i, j) = img1.at<Vec3b>(i, j);

  for(int i = 0; i < img2.rows; i++)
    for(int j = 0; j < img2.cols; j++)
      compare.at<Vec3b>(i, j + img1.cols) = img2.at<Vec3b>(i, j);

  for (size_t j = 0; j < feat2.size(); j ++) {
    vector<float> feat_vec(feat2[j].fea, feat2[j].fea + 64);

    const int knn = 2;
    vector<int> match_idx(knn);
    vector<float> match_dis(knn);
    fnn1.knnSearch(feat_vec, match_idx, match_dis, knn, flann::SearchParams());

    vector<float> verify_vec(feat1[match_idx[0]].fea, feat1[match_idx[0]].fea + 64);
    vector<int> verify_idx(knn);
    vector<float> verify_dis(knn);
    fnn2.knnSearch(verify_vec, verify_idx, verify_dis, knn, flann::SearchParams());
    // Lookup NN reversely to ensure the two are a match
    // Also eliminate the match if their y coordinates differs too much since we
    // are dealing with panorama
    if (verify_idx[0] == (int)j && abs(feat1[match_idx[0]].y - feat2[j].y) < 100) {
      match_pts.push_back(MatchPts(&feat1[match_idx[0]], &feat2[j]));

      int i = match_idx[0];
      int x1 = feat1[i].x, y1 = feat1[i].y;
      for(int di = -1; di <= 1; di++)
        for(int dj = -1; dj <= 1; dj++)
          img1_match.at<Vec3b>(y1+di, x1+dj) = Vec3b(0, 0, 255);

      int x2 = feat2[j].x, y2 = feat2[j].y;
      for(int di = -1; di <= 1; di++)
        for(int dj = -1; dj <= 1; dj++)
          img2_match.at<Vec3b>(y2+di, x2+dj) = Vec3b(0, 0, 255);

      Point p1(feat1[i].x, feat1[i].y);
      Point p2(feat2[j].x + img1.cols, feat2[j].y);
      line(compare, p1, p2, Scalar(0, 0, 255), 2);
    }
  }
  imwrite("compare.jpg", compare);
  imwrite("match1.jpg", img1_match);
  imwrite("match2.jpg", img2_match);

  // Compute optimal shift with RANSAC
  const int ITER = 1000;
  const int samples = 4;
  const float threshold = 5;

  int inlier_max = 0;
  float ret_x, ret_y;
  for (int iter = 0; iter < ITER; iter ++) {
    float offx = 0, offy = 0;
    for (int i = 0; i < samples; i ++) {
      int id = rand() % match_pts.size();
      offx += (match_pts[id].dsp1->x - match_pts[id].dsp2->x);
      offy += (match_pts[id].dsp1->y - match_pts[id].dsp2->y);
    }
    offx /= samples, offy /= samples;

    int inlier_cnt = 0;
    for (size_t i = 0; i < match_pts.size(); i ++) {
      float tx = (match_pts[i].dsp1->x - match_pts[i].dsp2->x);
      float ty = (match_pts[i].dsp1->y - match_pts[i].dsp2->y);

      float dis = hypot(offx - tx, offy - ty);
      inlier_cnt += (dis < threshold);
    }

    if (inlier_cnt > inlier_max)
      inlier_max = inlier_cnt, ret_x = offx, ret_y = offy;
  }

  printf("Best shift (%f, %f) with %d inliers out of %d matches.\n"
         , ret_x, ret_y, inlier_max, (int)match_pts.size());

  shift_x = (int)ret_x, shift_y = (int)ret_y;
}
