#include "MSOP.h"


static void showImg(const Mat& img) {
    double min, max;
    minMaxIdx(img, &min, &max);
    Mat imgS;
    convertScaleAbs(img, imgS, 255/max);
    resize(imgS, imgS, Size(0, 0), 0.5, 0.5);
    namedWindow("debug", WINDOW_AUTOSIZE);
    imshow("debug", imgS);
    waitKey(0);
}

static void writeImg(const Mat& img, const char* name) {
    double min, max;
    minMaxIdx(img, &min, &max);
    Mat imgS;
    convertScaleAbs(img, imgS, 255/max);
    imwrite(name, imgS);
}

static void writeImgPseudo(const Mat& img, const char* name) {
    double min, max;
    minMaxIdx(img, &min, &max);
    Mat imgS;
    convertScaleAbs(img, imgS, 255/max);
    Mat out;
    applyColorMap(imgS, out, COLORMAP_JET);
    imwrite(name, out);
}

MSOP::MSOP(int s, float sp, float sd, float si, int n):
 subRate(s), sigP(sp), sigD(sd), sigI(si), nPts(n) { }

// Convolve each layer of m with a Guassian kernel with
// vatriace = sig in place
void MSOP::tensorConv(vector<Mat>& m, float sig) {
    for (int i = 0; i < (int) m.size(); i++) {
        GaussianBlur(m[i], m[i], Size(0,0), sig, 0);
    }
}

// dir: 0 => x; 1 => y
void MSOP::calGrad(Mat& src, Mat& dst, int dir) {
    assert(src.type() == CV_32F);
    float kernel[] =  {-1, 0, 1};
    Mat ker;
    if (dir == 0)
        ker = Mat(1, 3, CV_32F, kernel);
    else
        ker = Mat(3, 1, CV_32F, kernel);
    filter2D(src, dst, CV_32F, ker);
}

void MSOP::outProd(vector<Mat>& Pl, vector<Mat>& Hl) {
    Hl[0] = Pl[0].mul(Pl[0]);
    Hl[1] = Pl[0].mul(Pl[1]);
    Hl[2] = Pl[0].mul(Pl[1]);
    Hl[3] = Pl[1].mul(Pl[1]);
}

// REVIEW: Check no seg. fault in second for loop
// Calculate corner strength and suppress low strength points
void MSOP::findCandidates(vector<Mat>& Hl, Mat& fHM, vector<Point>& pts) {
    cout << "Finding candidate points" << endl;
    int m = Hl[0].rows;
    int n = Hl[0].cols;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float a0 = Hl[0].at<float>(i, j);
            float a1 = Hl[1].at<float>(i, j);
            float a2 = Hl[2].at<float>(i, j);
            float a3 = Hl[3].at<float>(i, j);
            fHM.at<float>(i, j) = (a0*a3 - a1*a2) / (a0 + a3);
        }
    }
    // writeImgPseudo(fHM, "fhm_ps.png");

    // cout << fHM << endl;
    // Suppress corners
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float cenVal = fHM.at<float>(i, j);
            // cout << cenVal << endl;
            if (cenVal < 10 or cenVal != cenVal)
                continue;
            bool largest = true;
            for (int ii = -3; ii < 4; ii++) {
                for (int jj = -3; jj < 4; jj++) {
                    // Check boundaries
                    int y = ((ii + i) < 0) ? 0 : (ii + i);
                    y = (y > m) ? m : y;
                    int x = ((jj + j) < 0) ? 0 : (jj + j);
                    x = (x > n) ? n : x;
                    if (ii != 0 &&  jj != 0 && cenVal <= fHM.at<float>(y, x))
                        largest = false;
                }
            }
            if (largest) {
                pts.push_back(Point(j, i));
            }
        }
    }
    cout << "Find " << pts.size() << " interest points" << endl;
}

bool cornerSortF(const Corner& a, const Corner& b) {
    return (a.f > b.f);
}

bool cornerSortR(const Corner& a, const Corner& b) {
    return (a.r > b.r);
}

void MSOP::adapNonMaxSup(const Mat& f,
    const vector<Point>& pts, vector<Corner>& nmCor) {
    cout << "Applying adaptive non-max suppression" << endl;
    // Construct Corner vector for sorting
    vector<Corner> cors;
    cors.resize(pts.size());
    for (int i = 0; i < (int) pts.size(); i++) {
        Corner tmp;
        tmp.x = (float) pts[i].x;
        tmp.y = (float) pts[i].y;
        tmp.f = f.at<float>(pts[i].y, pts[i].x);
        cors[i] = tmp;
        // cout << tmp.x << " " << tmp.y << endl;
    }
    // Sort by f
    // REVIEW: Verify cors is decreasing
    std::sort(cors.begin(), cors.end(), cornerSortF);
    assert(cors[0].f >= cors[1].f);
    // distance to the closest point with a greater score
    float crobust = 0.9; // paper's setting
    for (int i = 0; i < (int) cors.size(); i++) {
        Corner& c = cors[i];
        float rmin = numeric_limits<float>::max();
        for (int j = 0; j < i; j++) {
            Corner& cj = cors[j];
            if (c.f < crobust*cj.f) {
                float dist = pow(c.x - cj.x, 2) + pow(c.y - cj.y ,2);
                dist = pow(dist, 0.5);
                if (dist < rmin)
                    rmin = dist;
            }
            // cout << i << " " << j << endl;
        }
        c.r = rmin;
        assert(cors[i].r == rmin);
    }
    // Sort by r
    // REVIEW: Does not consider equal values (r)
    std::sort(cors.begin(), cors.end(), cornerSortR);
    nPts = ((int) cors.size() > nPts)? nPts : cors.size();
    if ((int) nmCor.size() != nPts)
        nmCor.resize(nPts);
    std::copy(cors.begin(), cors.begin()+nPts, nmCor.begin());
    cout << "After suppression " << nmCor.size()\
        << " points remain, r = " << nmCor.back().r << endl;
}

void MSOP::subPixelRefine(const Mat& f, vector<Corner>& cor) {
    cout << "Performing sub-pixel refinement" << endl;
    for (int i = 0; i < (int) cor.size(); i++) {
        Corner& c = cor[i];
        int x = (int) round(c.x);
        int y = (int) round(c.y);
        float f00 = f.at<float>(y, x);
        // Compute first order derivative
        float fp[2];
        fp[0] = (f.at<float>(y, x+1) - f.at<float>(y, x-1)) / 2;
        fp[1] = (f.at<float>(y+1, x) - f.at<float>(y-1, x)) / 2;
        Mat fpMat = Mat(2, 1, CV_32F, fp);
        // Compute second order derivative
        float fpp[2][2];
        fpp[0][0] = (f.at<float>(y, x+1) + f.at<float>(y, x-1) - 2*f00);
        fpp[0][1] = (f.at<float>(y-1, x-1) - f.at<float>(y-1, x+1));
        fpp[0][1] = (fpp[0][1] + (f.at<float>(y+1, x+1) - f.at<float>(y+1, x-1)))/4;
        fpp[1][0] = fpp[0][1];
        fpp[1][1] = (f.at<float>(y+1, x) + f.at<float>(y-1, x) - 2*f00);
        Mat fppMat = Mat(2, 2, CV_32F, fpp);
        // Compute subpixel coor.
        Mat sub = -1.0 * (fppMat.inv() *  fpMat);
        // Save back
        c.x += sub.at<float>(0, 0);
        c.y += sub.at<float>(1, 0);
    }
}

void MSOP::calOrient(vector<Mat>& PlG, vector<Corner>& cor) {
    cout << "Calculating orientation" << endl;
    tensorConv(PlG, 4.5);
    for (int i = 0; i < (int) cor.size(); i++) {
        Corner& c = cor[i];
        int x = (int) round(c.x);
        int y = (int) round(c.y);
        float b = PlG[1].at<float>(y, x);
        float a = PlG[0].at<float>(y, x);
        c.o = acos(a / sqrt(pow(a, 2) + pow(b, 2)));
        // c.o = atan(PlG[1].at<float>(y, x) / PlG[0].at<float>(y, x));
    }
}

void MSOP::warp2Local(const Mat& glo, Mat& loc, float the, Point bias) {
    float rot[2][2] = {{cos(the), -sin(the)}, {sin(the), cos(the)}};
    Mat rotMat(2, 2, CV_32F, rot);
    loc = rotMat * glo;
    for (int i = 0; i < loc.cols; i++) {
        loc.at<float>(0, i) += bias.x;
        loc.at<float>(1, i) += bias.y;
    }
}

// Auume l is smaller than 1
void MSOP::calDescriptors(Mat& img, int mag, vector<Corner>& cor, vector<Descriptor>& dsps) {
    cout << "Calculating Descriptors..." << endl;
    // Generate desciptors for each corner
    Mat imgB;
    // blur image. Var fixed to 4.5
    GaussianBlur(img, imgB, Size(0,0), 4.5, 0);
    for (int i = 0; i < (int) cor.size(); i++) {
        Corner& c = cor[i];
        float the = c.o;
        Mat pts(2, 64, CV_32F);
        Mat cen(2, 1, CV_32F);
        int cnt = 0;
        for (int m = -17; m <= 18; m += 5) {
            for (int n = -17; n <= 18; n += 5) {
                pts.at<float>(0, cnt) = c.x + n;
                pts.at<float>(1, cnt) = c.y + m;
                cnt++;
            }
        }
        assert(cnt == 64);
        Mat ptsLocal;
        // REVIEW: Check the correctness of the angle
        warp2Local(pts, ptsLocal, -(M_PI/2 - the), Point(c.x, c.y));
        dsps.push_back(Descriptor());
        Descriptor& tmpDsp = dsps[dsps.size() - 1];
        // Convert to original scale
        tmpDsp.x = (int) round(c.x * mag);
        tmpDsp.y = (int) round(c.y * mag);
        for (int j = 0; j < 64; j++) {
            int ptX = (int) round(ptsLocal.at<float>(0, j));
            int ptY = (int) round(ptsLocal.at<float>(1, j));
            tmpDsp.fea[j] = img.at<float>(ptY, ptX);
        }
    }
}

// assume l is smaller than 1
void MSOP::drawCorners(const Mat& img, int mag, const vector<Corner>& cor, const char* name) {
    Mat canvas;
    img.copyTo(canvas);
    for (int i = 0; i < (int) cor.size(); i++) {
        const Corner& c = cor[i];
        float theta = c.o;
        int cenX = (int) round(c.x * mag);
        int cenY = (int) round(c.y * mag);
        Point cen = Point(cenX, cenY);
        // corners of the rect. in patch coor.
        float rc[2][4] = {{-20, 20, 20, -20}, {-20, -20, 20, 20}};
        Mat rectCor(2, 4, CV_32F, rc);
        rectCor = rectCor * mag;
        // corners of the rect. in image coor.
        Mat rectCorGlb(2, 4, CV_32F);
        warp2Local(rectCor, rectCorGlb, -(M_PI/2.0 - theta), cen);
        // Draw on canvas
        // four rectangle corners and four middle points
        vector<Point> cornPts(rectCorGlb.cols);
        vector<Point> midPts(rectCorGlb.cols);
        for (int j = 0; j < rectCor.cols; j++) {
            cornPts[j] = Point(rectCorGlb.at<float>(0, j), rectCorGlb.at<float>(1, j));
        }
        midPts[0] = (cornPts[0] + cornPts[1]) / 2.0;
        midPts[1] = (cornPts[1] + cornPts[2]) / 2.0;
        midPts[2] = (cornPts[2] + cornPts[3]) / 2.0;
        midPts[3] = (cornPts[3] + cornPts[0]) / 2.0;
        Point corrPts;
        for (int k = 0; k < rectCor.cols; k++) {
            Point midVec = midPts[k] - cen;
            float tmp = sqrt(pow(midVec.x, 2) + pow(midVec.y, 2));
            if (abs(acos(midVec.x / tmp) - theta) <= M_PI/4) {
                corrPts = midPts[k];
            }
        }
        // Draw lines
        line(canvas, cornPts[0], cornPts[1], Scalar(255, 255, 255), 2);
        line(canvas, cornPts[1], cornPts[2], Scalar(255, 255, 255), 2);
        line(canvas, cornPts[2], cornPts[3], Scalar(255, 255, 255), 2);
        line(canvas, cornPts[3], cornPts[0], Scalar(255, 255, 255), 2);
        line(canvas, cen, corrPts, Scalar(255, 255, 255), 2);
        // Draw center
        circle(canvas, cen, 1, Scalar(255, 255, 255));
    }
    imwrite(name, canvas);
    cout << "Corners are written to image" << endl;
}

// void MSOP::writePoints(const Mat& img, vector<Point>& pts, int mag, const char* fileName) {
//     Mat imgC;
//     img.copyTo(imgC);
//     for (int i = 0; i < (int) pts.size(); i++) {
//         int glbX = pts[i].x * mag;
//         int glbY = pts[i].y * mag;
//         circle(imgC, Point(glbX, glbY), 1, Scalar(255, 0, 0));
//     }
//     imwrite(string(fileName), imgC);
// }

template<typename T>
void writePoints(const Mat& img, vector<T>& pts, int mag, const char* fileName) {
    Mat imgC;
    img.copyTo(imgC);
    for (int i = 0; i < (int) pts.size(); i++) {
        int glbX = (int) pts[i].x * mag;
        int glbY = (int) pts[i].y * mag;
        circle(imgC, Point(glbX, glbY), 1, Scalar(255, 0, 0));
    }
    imwrite(string(fileName), imgC);
}

void MSOP::extract(Mat img) {
    // Convert image to gray scale and float

    // REVIEW: Check image order
    cvtColor(img, img, COLOR_BGR2GRAY);
    img.convertTo(img, CV_32F);

    // Build Gaussian pyramid
    vector<Mat> imgPr;
    float minScale = 0.1;
    Mat imgDown = img;
    imgPr.push_back(img);
    while (1) {
        // Size(0, 0) => size is determined by sigP.
        GaussianBlur(imgDown, imgDown, Size(0, 0), sigP, 0);
        resize(imgDown, imgDown, Size(0, 0), 1.0/subRate, 1.0/subRate);
        if ((float) imgDown.rows / img.rows < minScale)
            break;
        if (imgDown.rows < 200 || imgDown.cols < 200)
            break;
        imgPr.push_back(imgDown);
    }
    cout << imgPr.size() << " pyramids built. " \
    << "min size: " << imgPr.back().rows << ", "\
    << imgPr.back().cols << endl;

    // Compure Harris matrix
    cout << "Computing Harris matrix" << endl;
    for (int l = 1; l < 2; l++) {
    // for (int l = 0; l < (int) imgPr.size(); l++) {
        Mat pImg = imgPr[l];
        vector<Mat> Hl, PlG, PlGCopy;
        Hl.resize(4);
        PlG.resize(2);
        PlGCopy.resize(2);
        // Calculate gradients
        calGrad(pImg, PlG[0], 0); // x-direction
        calGrad(pImg, PlG[1], 1); // y-direction
        PlG[0].copyTo(PlGCopy[0]); // deep copy
        PlG[1].copyTo(PlGCopy[1]); // REVIEW: Check no need to pre-allocate
        tensorConv(PlG, sigD);
        outProd(PlG, Hl); // ourter product
        tensorConv(Hl, sigI);
        // Collect candidate corners
        Mat fHM = Mat::zeros(pImg.rows, pImg.cols, CV_32F);
        vector<Point> pts;
        findCandidates(Hl, fHM, pts);
        writePoints<Point>(imgPr[0], pts, pow(2, l), "corner.png");
        // Apply adaptive non-max suppression
        vector<Corner> cor(nPts);
        adapNonMaxSup(fHM, pts, cor);
        writePoints<Corner>(imgPr[0], cor, pow(2, l), "corner_sup.png");
        subPixelRefine(fHM, cor);
        writePoints<Corner>(imgPr[0], cor, pow(2, l), "corner_refined.png");
        calOrient(PlGCopy, cor);
        drawCorners(imgPr[0], pow(2, l), cor, "corner_window.png");
        // Generate desciptors from Corners
    }
}