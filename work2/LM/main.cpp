#include <iostream>
#include <opencv/cv.hpp>
#include <math.h>
#include <iomanip>

using namespace std;
using namespace cv;

#define N 40

double x_data[] = {0.,0.05128205,0.1025641,0.15384615,0.20512821,0.25641026,
                  0.30769231,0.35897436,0.41025641,0.46153846,0.51282051,0.56410256,
                  0.61538462,0.66666667,0.71794872,0.76923077,0.82051282,0.87179487,
                  0.92307692,0.97435897,1.02564103,1.07692308,1.12820513,1.17948718,
                  1.23076923,1.28205128,1.33333333,1.38461538,1.43589744,1.48717949,
                  1.53846154,1.58974359,1.64102564,1.69230769,1.74358974,1.79487179,
                  1.84615385,1.8974359,1.94871795,2.        };



double y_data[] = {  50.14205309,74.52945195,-125.60406092,50.73876215,-78.06990572,
                    69.07600961,72.59307969,-89.85886577,49.63661953,-116.99899809,
                    75.61742457,54.64519908,-47.2762507,53.40895121,-143.71858796,
                    72.98163061,26.37395524,-2.60377048,65.1767561,-148.47687307,
                    64.41096274,-10.09721052,35.12977766,71.16446338,-138.68616457,
                    55.83174324,-55.22601629,61.38557638,75.7080723,-109.84633432,
                    48.68186317,-95.74894683,71.69891297,66.86944822,-70.12822282,
                    52.36352661,-130.17602848,74.33260689,44.69661749,-26.65394757};

double func(double beta[], double x)
{
    double a = beta[0];
    double b = beta[1];
    return a*cos(b*x) + b*sin(a*x);
}

Mat f(double beta[])
{
    Mat f(N, 1, CV_64FC1);
    for(int i = 0; i < N; i++)
        f.at<double>(i, 0) = func(beta, x_data[i]) - y_data[i];
    return f;
}

double F(double beta[]) // calc mse
{
    double err = 0.0;
    for(int i = 0; i < N; i++)
    {
        err += pow((func(beta, x_data[i]) - y_data[i]), 2);
    }
    return err / double(N);
}

double Derive(int i, int j, double beta[])
{
    double beta_copy_0[2];
    double beta_copy_1[2];
    for(int k = 0; k < 2; k++)
    {
        if(j == k)
        {
            beta_copy_0[k] = beta[k] + 0.00001;
            beta_copy_1[k] = beta[k] - 0.00001;
        }
        else
        {
            beta_copy_0[k] = beta[k];
            beta_copy_1[k] = beta[k];
        }
    }
    return (func(beta_copy_0, x_data[i]) - func(beta_copy_1, x_data[i])) / 0.00002;
}

Mat Jacobian(double beta[])
{
    Mat J(N, 2, CV_64FC1);
    for(int j = 0; j < 2; j++)
    {
        for (int i = 0; i < N; i++)
        {
            J.at<double>(i,j) = Derive(i, j, beta);
        }
    }
    return J;
}

int main()
{
    Mat x = Mat(1, N, CV_64FC1, x_data);
    Mat y = Mat(1, N, CV_64FC1, y_data);

    double beta[2] = {52, 99}; // initial

    double u = 1;
    double v = 2;

    int step = 500;
    while(step--)
    {
        cout << "step:" << step << endl;
        Mat J = Jacobian(beta);
        Mat H = J.t()*J + u*Mat::eye(2,2, CV_64FC1);
        Mat delta = -H.inv() * J.t() * f(beta);

        double beta_new[2] = {beta[0] + delta.at<double>(0,0), beta[1] + delta.at<double>(1,0)};
        cout << beta_new[0] << ", " << beta_new[1] << endl;

        double Fx = F(beta);
        double Fnew = F(beta_new);

        Mat tmp = delta.t()*(u*delta - J.t()*f(beta));
        double rho = (Fx - Fnew) / tmp.at<double>(0,0);

        if(rho > 0)
        {
            v = 2;
            beta[0] = beta_new[0];
            beta[1] = beta_new[1];
            u *= max(double(1/3), double(1-pow(2*rho-1, 3)));
        }
        else
        {
            u *= v;
            v *= 2;
        }
    }


    cout << beta[0] << ", " << beta[1] << endl;


    return 0;
}
