#include <iostream>
#include <opencv/cv.hpp>
#include <utility>
#include <iomanip>

#include "lm.h"

#define SIZE 5 // legth of block
#define ATTEMP 1 // attempts for LM initial

using namespace std;
using namespace cv;

vector<Point2f> findCorners(Mat& img)
{
    Mat img_gr, img_th, img_di;

    if (img.channels() != 1)
    {
        cvtColor(img, img_gr, COLOR_BGR2GRAY);
    }
    threshold(img_gr, img_th, 0, 255, THRESH_OTSU);
    dilate(img_th, img_di, Mat());

    // rectangle(img_di, Point(0,0), Point(img_di.cols-1, img_di.rows-1), Scalar(255,255,255), 3, LINE_8);

    vector<vector<Point>> contours;
    vector<vector<Point>> rectangles;
    vector<Vec4i> hierarchy;
    findContours(img_di, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    // drawContours(img, rectangles, -1, Scalar(255,0,0), 2);

    for(uint i = 0; i < contours.size(); i++)
    {
        vector<Point>& contour = contours[i];
        vector<Point> approx_contour;
        approxPolyDP(contour, approx_contour, 2.0, true);
        if(approx_contour.size() == 4)
            rectangles.push_back(approx_contour);
    }

    vector<Point> rectangle;
    vector<Point> rectangle_cmp;

    vector<Point2f> corners; // 80
    bool finded[50][4] = {false};

    for(uint i = 0; i < rectangles.size(); i++)
    {
        rectangle = rectangles[i];
        for(uint j = 0; j < rectangles.size(); j++)
        {
            if(j == i)
                break;
            rectangle_cmp =rectangles[j];
            for(uint m = 0; m < 4; m++)
            {
                if(finded[i][m])
                    continue;
                for(uint n = 0; n < 4; n++)
                {
                    float dist = norm(rectangle[m] - rectangle_cmp[n]);
                    if(dist < 8.0)
                    {
                        finded[i][m] = true;
                        finded[j][n] = true;
                        Point2f corner = Point2f(rectangle[m] + rectangle_cmp[n]) / 2.0;
                        corners.push_back(corner);
                    }
                }
            }
        }
    }
    TermCriteria criteria = TermCriteria(
                CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                40, //maxCount=40
                0.001 );  //epsilon=0.001
    cornerSubPix(img_gr, corners, Size(2,2), Size(-1,-1), criteria); // when corners in vector<Point2d>, error 'assert count>=0' occur!  why??
    return corners;
}

Mat get_element(Point2f p, Point3f P)
{
    double x = P.x;
    double y = P.y;
    double z = P.z;

    double u = p.x;
    double v = p.y;

    Mat ret = (Mat_<double>(2, 12) << x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -m_34*u,
               0, 0, 0, 0, x, y, z , 1, -v*x, -v*y, -v*z, -m_34*v);
    return ret;
}

void sort_corners(vector<Point2f>& corners)
{
    sort(corners.begin(), corners.end(), [](Point2f a, Point2f b) -> bool {return a.x < b.x; });
    for(uint i = 0; i < 10; i++)
    {
        sort(corners.begin() + 8*i, corners.begin() + 8*i + 8, [](Point2f a, Point2f b) -> bool {return a.y > b.y; });
    }
}

vector<Point3f> get_word_coordinate()
{
    vector<Point3f> word_coordinate;
    vector<pair<int, int>> xy_vec = {pair<int, int>(4*SIZE, 0), pair<int, int>(3*SIZE, 0), pair<int, int>(2*SIZE, 0), pair<int, int>(SIZE, 0), pair<int, int>(0, 0),
                                     pair<int, int>(0, SIZE), pair<int, int>(0, 2*SIZE), pair<int, int>(0, 3*SIZE), pair<int, int>(0, 4*SIZE), pair<int, int>(0, 5*SIZE)};
    for(auto& xy : xy_vec)
    {
        for(uint i = 1; i < 9; i++)
            word_coordinate.push_back(Point3f(xy.first, xy.second, SIZE*i));
    }
    return word_coordinate;
}

pair<Mat, double> calc_m(int idx[], vector<Point2f> corners, vector<Point3f> word_coordinate)
{
    Mat P, b, m;
    Mat element_0 = get_element(corners[idx[0]], word_coordinate[idx[0]]);
    Mat element_1 = get_element(corners[idx[1]], word_coordinate[idx[1]]);
    Mat element_2 = get_element(corners[idx[2]], word_coordinate[idx[2]]);
    Mat element_3 = get_element(corners[idx[3]], word_coordinate[idx[3]]);
    Mat element_4 = get_element(corners[idx[4]], word_coordinate[idx[4]]);
    Mat element_5 = get_element(corners[idx[5]], word_coordinate[idx[5]]);

    vector<Mat>P_data = {element_0.colRange(0, 11), element_1.colRange(0, 11), element_2.colRange(0, 11), element_3.colRange(0, 11), element_4.colRange(0, 11), element_5.colRange(0, 11)};
    vector<Mat>b_data = {-element_0.colRange(11, 12), -element_1.colRange(11, 12), -element_2.colRange(11, 12), -element_3.colRange(11, 12), -element_4.colRange(11, 12), -element_5.colRange(11, 12)};

    vconcat(P_data, P);
    vconcat(b_data, b);

    P = P.rowRange(0, 11);
    b = b.rowRange(0, 11);

    m = P.inv() * b;



    LM optimizer = LM(m);
    double err = optimizer.run(word_coordinate, corners);
    Mat optimized_m = optimizer.get_result();

    return pair<Mat, double>(optimized_m, err);
}


int main()
{
    Mat img = imread("../hw4_2.png");
    if(img.empty())
        throw runtime_error("Img can not find!");

    namedWindow("img", WINDOW_KEEPRATIO);

    vector<Point2f> corners;
    corners = findCorners(img);
    sort_corners(corners);

    vector<Point3f> word_coordinate = get_word_coordinate();

    for(uint i = 0; i < corners.size(); i++)
    {
        // cout << word_coordinate.at(i) << endl;
        circle(img, corners[i], 3, Scalar(0,0,255), 1);
        // imshow("img", img);
        // waitKey(0);
    }



    RNG rng(2);
    int idx[6];
    double min_err = 1e10;
    pair<Mat, double> optim;
    for(int j = 0; j < ATTEMP; j++)
    {
        for(int i = 0; i < 6; i++)
            idx[i] = rng.uniform(0,80);
        // TODO make sure stddev of array idx big enough before calculate m
        pair<Mat, double> tmp = calc_m(idx, corners, word_coordinate);
        if(tmp.second < min_err)
        {
            min_err = tmp.second;
            optim = tmp;
        }
    }
    Mat m = optim.first;
    Mat M;
    Mat m34 =(Mat_<double>(1, 1) << m_34*1);
    vconcat(m, m34, m);
    m /= norm(m);
    M = m.reshape(1, 3); // reshape(channel, row)
    cout << "----------M----------" << endl;
    cout << M.type() << endl;
    cout << CV_64FC1 << endl;


    Mat A = M.colRange(0,3);
    Mat a1 = A.rowRange(0,1);
    Mat a2 = A.rowRange(1,2);
    Mat a3 = A.rowRange(2,3);
    Mat b = M.colRange(3,4);

    double rho = 1.0 / norm(a3);
    Mat r3 = rho*a3;

    Mat u0_mat = rho*rho*a1*a3.t();
    Mat v0_mat = rho*rho*a2*a3.t();
    double u0 = u0_mat.at<double>(0,0);
    double v0 = v0_mat.at<double>(0,0);
    Mat theta_tmp = -(a1.cross(a3))*(a2.cross(a3)).t() / norm(a1.cross(a3)) * norm(a2.cross(a3));
    double theta = acos(theta_tmp.at<double>(0,0));

    double alpha = rho*rho * norm(a1.cross(a3)) * sin(theta);
    double beta = rho*rho * norm(a2.cross(a3)) * sin(theta);
    Mat r1 = a2.cross(a3) / norm(a2.cross(a3));
    Mat r2 = r3.cross(r1);

    Mat K = (Mat_<double>(3,3) << alpha, -alpha/tan(theta), u0, 0, beta / sin(theta), v0, 0, 0, 1);

    Mat t = rho*K.inv()*b;

    Mat R, RT;
    vector<Mat> R_data = {r1,r2, r3};
    vconcat(R_data, R);
    vector<Mat> RT_data = {R,t};

    hconcat(RT_data, RT);
    cout << RT.size << endl;

    cout << "rho*M:" << endl;
    cout << rho*M << endl;
    cout << "K*RT" << endl;
    cout << K*RT << endl;

    for(int i = 0; i < 80; i++)
    {
        cout << "Point " << i << endl;
        Point3f point = word_coordinate[i];
        Mat P = (Mat_<double>(4,1) << point.x, point.y, point.z, 1);
        Mat p = K*RT*P;
        p /= p.at<double>(0,2);

        cout << "reprejection coordinate:\n" << p.t() << endl;
        //cout << "world coordinate:\n" << word_coordinate[i] << endl;
        cout << "detected coordinate:\n" << corners[i] << '\n' << endl;
    }


    cout << "rho*M:" << endl;
    cout << rho*M << endl;
    cout << "K*RT" << endl;
    cout << K*RT << endl;
    cout << "RT" << endl;
    cout << RT << endl;
    cout << "K" << endl;
    cout << K << endl;

    imshow("img", img);

    waitKey(0);
    return 0;
}
