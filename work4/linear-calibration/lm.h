#ifndef LM_H
#define LM_H
#include<opencv/cv.hpp>
#include<vector>


#define m_34 1 // doesn't affect result

using namespace cv;
using namespace std;

class LM
{
public:
    LM(Mat init_m): m(init_m){}
    double run(vector<Point3f> P_vec, vector<Point2f> p_vec);
    Mat get_result(){return m;}
    Mat get_mat(Point3f P, Point2f p);
private:
    Mat Jacobian(Mat beta);
    double Func(Mat beta, Mat data);
    double Derive(Mat beta, Mat data, int n);


    Mat m;
};

#endif // LM_H
