#include "lm.h"
#include <iostream>

Mat LM::get_mat(Point3f P, Point2f p)
{
    double x = P.x;
    double y = P.y;
    double z = P.z;

    double u = p.x;
    double v = p.y;

    return (Mat_<double>(2, 12) << x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -m_34*u,
             0, 0, 0, 0, x, y, z , 1, -v*x, -v*y, -v*z, -m_34*v);
}

double LM::Func(Mat m, Mat data) // calc mse
{
    Mat P = data.colRange(0,11); // 2n*11
    Mat b = - data.colRange(11, 12); // 2n*1
    double err = sum(b - P*m).val[0];
    return err;
}

double LM::Derive(Mat m, Mat data, int n)
{
    Mat m1, m2;
    m.copyTo(m1);
    m.copyTo(m2);
    m1.at<double>(n,0) -= 0.000001;
    m2.at<double>(n,0) += 0.000001;
    double p1 = Func(m1, data);
    double p2 = Func(m2, data);
    double d = (p2-p1) / 0.000002;
    return d;
}


double LM::run(vector<Point3f> P_vec, vector<Point2f> p_vec)
{
    uint N = P_vec.size();
    int step = 500;
    Mat J = Mat(N, 11, CV_64FC1);
    Mat fx = Mat(N, 1, CV_64FC1);
    Mat fx_tmp = Mat(N, 1, CV_64FC1);
    Mat xk = m.rowRange(0,11);
    Mat xk_tmp = xk.clone();

    int u = 1, v= 2;
    double mse = 0.0, mse_tmp = 0.0;

    while(step--)
    {
        for(uint i = 0; i < N; i++)
        {
            Mat data = get_mat(P_vec.at(i), p_vec.at(i));
            fx.at<double>(i, 0) = Func(xk, data);
            mse += pow(fx.at<double>(i,0), 2);
            for(int j = 0; j < 11; j++)
                J.at<double>(i, j) = Derive(xk, data, j);
        }
        mse /= double(N);


        Mat H = J.t() * J + u*Mat::eye(11,11,CV_64FC1);
        Mat dx = -H.inv() * J.t() * fx;

        xk.copyTo(xk_tmp);
        xk_tmp += dx;

        for(uint i = 0; i < N; i++)
        {
            Mat data = get_mat(P_vec.at(i), p_vec.at(i));
            fx_tmp.at<double>(i, 0) = Func(xk_tmp, data);
            mse_tmp += pow(fx_tmp.at<double>(i,0),2);
        }
        mse_tmp /= double(N);

        if((mse - mse_tmp) < 0.000000001)
            break;

        Mat tmp = dx.t()*(u*dx - J.t()*fx);
        double rho = (mse - mse_tmp) / tmp.at<double>(0,0);

        if(rho > 0)
        {
            v = 2;
            mse = mse_tmp;

            xk = xk_tmp;
            u *= max(double(1.0/3), double(1-pow(2*rho-1, 3)));
        }
        else
        {
            u *= v;
            v *= 2;
        }

    }
    m = xk;
    return mse;
}

