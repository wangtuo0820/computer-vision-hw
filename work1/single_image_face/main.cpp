#include <iostream>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat image, image_gray;      //定义两个Mat变量，用于存储每一帧的图像

    image = imread("/home/tuo/data/face.png");
    imshow("原图", image);

    cvtColor(image, image_gray, CV_BGR2GRAY);//转为灰度图
    equalizeHist(image_gray, image_gray);//直方图均衡化，增加对比度方便处理

    CascadeClassifier eye_Classifier;  //载入分类器
    CascadeClassifier face_cascade;    //载入分类器

    //加载分类训练器，OpenCv官方文档提供的xml文档，可以直接调用
    //xml文档路径  opencv\sources\data\haarcascades
    if (!eye_Classifier.load("/home/tuo/data/haarcascade_eye.xml"))  //需要将xml文档放在自己指定的路径下
    {
        cout << "Load haarcascade_eye.xml failed!" << endl;
        return 0;
    }

    if (!face_cascade.load("/home/tuo/data/haarcascade_frontalface_alt.xml"))
    {
        cout << "Load haarcascade_frontalface_alt failed!" << endl;
        return 0;
    }

    //vector 是个类模板 需要提供明确的模板实参 vector<Rect>则是个确定的类 模板的实例化
    vector<Rect> eyeRect;
    vector<Rect> faceRect;

    RNG rng;

    //检测关于眼睛部位位置
    eye_Classifier.detectMultiScale(image_gray, eyeRect, 1.1, 4, 0 , Size(30, 30));
    for (size_t eyeIdx = 0; eyeIdx < eyeRect.size(); eyeIdx++)
    {
        rectangle(image, eyeRect[eyeIdx], Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)), 5);   //用矩形画出检测到的位置
    }

    //检测关于脸部位置
    face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 2, 0, Size(30, 30));
    for (size_t i = 0; i < faceRect.size(); i++)
    {
        rectangle(image, faceRect[i], Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)), 5);      //用矩形画出检测到的位置
    }

    imshow("人脸识别图", image);         //显示当前帧
    waitKey(0);

    return 0;
}
