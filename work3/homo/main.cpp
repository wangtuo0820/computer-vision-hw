#include <iostream>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

Mat get_matrix(Point2f& src, Point2f& dst)
{
    cout << src << endl;
    cout << dst << endl;

    float x = src.x;
    float y = src.y;
    float x_ = dst.x;
    float y_ = dst.y;
    float data[2][9] = {
        {x,y,1,0,0,0,-x*x_, -x_*y, -x_},
        {0,0,0,x,y,1,-y_*x, -y_*y, -y_}
    };
    for(int i=0; i < 2; i++)
    {
        for(int j=0; j<9; j++)
            cout << data[i][j] << " ";
        cout << endl;
    }
    return Mat(2,9,CV_32FC1, data);
}

int main()
{
    Mat img1 = imread("../img1.jpg");
    Mat img2 = imread("../img2.jpg");
    vector<Point2f> corners1;
    findChessboardCorners(img1, Size(9,6), corners1);
    for(auto p : corners1)
        circle(img1, p, 5, Scalar(255,0,0), -1);

    vector<Point2f> corners2;
    findChessboardCorners(img2, Size(9,6), corners2);
    for(auto p : corners2)
        circle(img2, p, 5, Scalar(255,0,0), -1);

    // imshow("img1", img1);
    // imshow("img2", img2);
#if 1
    Mat homo = findHomography(corners1, corners2);
    cout << homo << endl;
#else
    Mat test = Mat(2,9,CV_32FC1);
    test = get_matrix(corners1[0], corners2[0]);

    cout << CV_32FC1 << endl;
    cout << test.type() << endl;

    cout << test << endl;
#endif
    waitKey();
    return 0;
}
