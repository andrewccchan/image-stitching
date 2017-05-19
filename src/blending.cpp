#include "blending.h"
using namespace cv;

bool emptyPixel(Vec3b p) {
  // specific to images with 3 channels
  return p[0] == 0 && p[1] == 0 && p[2] == 0;
}

void directBlending(vector<Mat> &img, vector<int> &offx, vector<int> &offy) {
  int width = 0, height = 0;
  for (int i = 0; i < (int)img.size(); i ++) {
    width = max(width, offx[i] + img[i].cols);
    height = max(height, offy[i] + img[i].rows);
  }

  Mat combined = Mat::zeros(height, width, CV_8UC3);

  // for (int k = 0; k < (int)img.size(); k ++) {
  for (int k = (int)img.size() - 1; k >= 0; k --) {
    for(int i = 0; i < img[k].rows; i++)
      for(int j = 0; j < img[k].cols; j++) {
        if (emptyPixel(img[k].at<Vec3b>(i, j))) continue ;
        combined.at<Vec3b>(i + offy[k], j + offx[k]) = img[k].at<Vec3b>(i, j);
      }
  }

  imwrite("combined.jpg", combined);
}

void alphaBlending(vector<Mat> &img, vector<int> &offx, vector<int> &offy) {
  int width = 0, height = 0;
  for (int i = 0; i < (int)img.size(); i ++) {
    width = max(width, offx[i] + img[i].cols);
    height = max(height, offy[i] + img[i].rows);
  }

  Mat accu = Mat::zeros(height, width, CV_32FC3);
  Mat aw = Mat::zeros(height, width, CV_32F);
  Mat combined = Mat::zeros(height, width, CV_32FC3);

  for (int k = 0; k < (int)img.size(); k ++) {
    for(int i = 0; i < img[k].rows; i++)
      for(int j = 0; j < img[k].cols; j++) {
        if (emptyPixel(img[k].at<Vec3b>(i, j))) continue ;
        float weight = min(j + 1, img[k].cols - j);
        accu.at<Vec3f>(i + offy[k], j + offx[k]) += Vec3f(img[k].at<Vec3b>(i, j)) * weight;
        aw.at<float>(i + offy[k], j + offx[k]) += weight;
      }
  }
  for (int i = 0; i < height; i ++) {
    for (int j = 0; j < width; j ++) {
      if (aw.at<float>(i, j) == 0) combined.at<Vec3f>(i, j) = Vec3f(0, 0, 0);
      else combined.at<Vec3f>(i, j) = accu.at<Vec3f>(i, j) / aw.at<float>(i, j);
    }
  }
  combined.convertTo(combined, CV_8UC3);

  imwrite("combined.jpg", combined);
}

typedef pair<int, int> ii;
typedef pair<ii, float> iii;
void poissonBlending(vector<Mat> &img, vector<int> &offx, vector<int> &offy) {
  int width = 0, height = 0;
  for (int i = 0; i < (int)img.size(); i ++) {
    width = max(width, offx[i] + img[i].cols);
    height = max(height, offy[i] + img[i].rows);
  }

  vector<Mat> imgf(img.size());
  for (int i = 0; i < (int)img.size(); i ++) {
    img[i].convertTo(imgf[i], CV_32FC3);
  }

  Mat combined = Mat::zeros(height, width, CV_32FC3);
  Mat tmpMat = Mat::zeros(height, width, CV_32S); // for temporary tags

  static const int dx[4] = {0, 0, 1, -1};
  static const int dy[4] = {1, -1, 0, 0};

  // Place the first image directly onto the result
  for(int i = 0; i < imgf[0].rows; i++)
    for(int j = 0; j < imgf[0].cols; j++)
      if (!emptyPixel(imgf[0].at<Vec3f>(i, j)))
        combined.at<Vec3f>(i + offy[0], j + offx[0]) = imgf[0].at<Vec3f>(i, j);

  auto isBoundary = [&](int imgId, int i, int j) {
    // returns true if the point (j, i) is on the boundary of the original image
    if (i == 0 || i == imgf[imgId].rows - 1 || j == 0 || j == imgf[imgId].cols - 1) return true;

    for (int k = 0; k < 4; k ++) {
      int ti = i + dy[k];
      int tj = j + dx[k];
      if (emptyPixel(imgf[imgId].at<Vec3f>(ti, tj))) return true;
    }
    return false;
  };

  // Blend the other images one at a time
  for (int imgId = 1; imgId < (int)imgf.size(); imgId ++) {
    vector<ii> omega; // the cooridinate of interior points

    for(int i = 0; i < imgf[imgId].rows; i++)
      for(int j = 0; j < imgf[imgId].cols; j++) {
        if (!emptyPixel(combined.at<Vec3f>(i + offy[imgId], j + offx[imgId]))) {
          // already rendered, do nothing
        } else if (isBoundary(imgId, i, j)) {
          combined.at<Vec3f>(i + offy[imgId], j + offx[imgId]) = imgf[imgId].at<Vec3f>(i, j);
        } else {
          tmpMat.at<int>(i + offy[imgId], j + offx[imgId]) = omega.size();
          omega.push_back(ii(i + offy[imgId], j + offx[imgId]));
        }
      }

    vector<iii> edges; // store the sparse entries of A. (weight, (idx1, idx2))
    // construct the sparse matrix A
    for (int k = 0; k < (int)omega.size(); k ++) {
      int i = omega[k].first;
      int j = omega[k].second;

      edges.push_back(iii(ii(k, k), 4));
      for (int d = 0; d < 4; d ++) {
        int ti = i + dy[d];
        int tj = j + dx[d];
        if (emptyPixel(combined.at<Vec3f>(ti, tj))) {
          int tid = tmpMat.at<int>(ti, tj);
          edges.push_back(iii(ii(k, tid), -1));
        }
      }
    }

    // Solve with conjugate gradient method since the linear system is
    // symmetric, PD, and sparse

    // the "r0" in the conjugate gradient method, initialized to b - Ax
    Mat r = Mat::zeros(omega.size(), 1, CV_32FC3);
    // initialize r
    for (int k = 0; k < (int)omega.size(); k ++) {
      int i = omega[k].first;
      int j = omega[k].second;
      for (int d = 0; d < 4; d ++) {
        int ti = i + dy[d];
        int tj = j + dx[d];

        // TODO: check correctness
        r.at<Vec3f>(k) += combined.at<Vec3f>(ti, tj);
        r.at<Vec3f>(k) += imgf[imgId].at<Vec3f>(i - offy[imgId], j - offx[imgId]) -
                          imgf[imgId].at<Vec3f>(ti - offy[imgId], tj - offx[imgId]);
      }
    }
    Mat p = r.clone();

    auto dotProduct = [&](Mat &m1, Mat &m2) -> Vec3f {
      Vec3f ret(0, 0, 0);
      for (int i = 0; i < m1.rows; i ++) {
        ret += m1.at<Vec3f>(i).mul(m2.at<Vec3f>(i));
      }
      return ret;
    };

    // start iterations
    const int iters = 1500;
    Vec3f lst_inner = dotProduct(r, r);
    for (int iter = 0; iter < iters; iter ++) {
      Vec3f coeff(0, 0, 0);
      Mat Ap = Mat::zeros(omega.size(), 1, CV_32FC3);
      for (auto &&edge : edges) {
        int i = edge.first.first;
        int j = edge.first.second;
        float c = edge.second;
        Ap.at<Vec3f>(i) += p.at<Vec3f>(j) * c;
      }

      for (int k = 0; k < (int)omega.size(); k ++) {
        coeff += p.at<Vec3f>(k).mul(Ap.at<Vec3f>(k));
      }

      Vec3f alpha;
      divide(lst_inner, coeff, alpha);

      for (int k = 0; k < (int)omega.size(); k ++) {
        int i = omega[k].first;
        int j = omega[k].second;
        combined.at<Vec3f>(i, j) += p.at<Vec3f>(k).mul(alpha);
        r.at<Vec3f>(k) -= Ap.at<Vec3f>(k).mul(alpha);
      }

      Vec3f inner = dotProduct(r, r);
      Vec3f beta;
      divide(inner, lst_inner, beta);

      for (int k = 0; k < (int)omega.size(); k ++) {
        p.at<Vec3f>(k) = r.at<Vec3f>(k) + beta.mul(p.at<Vec3f>(k));
      }
      cout << "Iter " << iter << " Error: " << inner << endl;

      lst_inner = inner;

      if (iter % 150 == 0) {
        Mat outM;
        combined.convertTo(outM, CV_8UC3);
        string outn = "com" + to_string(imgId) + "_" + to_string(iter) + ".jpg";
        imwrite(outn, outM);
      }

      if (inner[0] < 10 && inner[1] < 10 && inner[2] < 10) break;
    }
  }
  // for (int i = 0; i < height; i ++) {
    // for (int j = 0; j < width; j ++) {
      // if (aw.at<float>(i, j) == 0) combined.at<Vec3f>(i, j) = Vec3f(0, 0, 0);
      // else combined.at<Vec3f>(i, j) = accu.at<Vec3f>(i, j) / aw.at<float>(i, j);
    // }
  // }
  combined.convertTo(combined, CV_8UC3);

  imwrite("combined.jpg", combined);
}
