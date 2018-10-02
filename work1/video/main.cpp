#include <iostream>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

int main()
{
    cout << "Press esc to quit." << endl;
    cout << "Press b to switch gray and color." << endl;
    cout << "Press + to scale up image." << endl;
    cout << "Press - to scale down image." << endl;

    VideoCapture capture(0);
    Mat frame, bw;

    if(!capture.isOpened())
        cout << "fail to open" << endl;

    namedWindow("demo");

    bool isGray = false;
    float rate = 1.0;

    while(true)
    {
        if(!capture.read(frame)){
            cout << "No Frame" << endl;
        }

        Size size = Size(640*rate, 480*rate);

        resize(frame, frame, size);
        cvtColor(frame, bw, COLOR_BGR2GRAY);

        if(isGray)
            imshow("demo", bw);
        else
            imshow("demo", frame);

        equalizeHist(bw, bw); // equalize histogram to add contrast

        CascadeClassifier eye_classifer;
        CascadeClassifier face_classifer;

        if(!eye_classifer.load("/home/tuo/data/haarcascade_eye_tree_eyeglasses.xml"))
        {
            cout << "Load haarcascade_eye_tree_eyeglasses.xml failed!" << endl;
            return 0;
        }
        if(!face_classifer.load("/home/tuo/data/haarcascade_frontalface_alt.xml"))
        {
            cout << "Load haarcascade_frontalface_alt failed!" << endl;
        }

        vector<Rect> eyeRect;
        vector<Rect> faceRect;

        RNG rng; // for generate random color rect
        eye_classifer.detectMultiScale(bw, eyeRect, 1.08, 6);
        for(size_t i = 0; i < eyeRect.size(); i++)
        {
            rectangle(frame, eyeRect[i], Scalar(0, 0, 0), 5);
        }

        face_classifer.detectMultiScale(bw, faceRect, 1.1, 4);
        for(size_t i = 0; i < faceRect.size(); i++)
        {
            rectangle(frame, faceRect[i], Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)), 5);
        }

        imshow("result", frame);

        char key = waitKey(1);
        if(key == 27)
        {
            cout << "exiting..." << endl;
            break;
        }
        switch (key) {
        case 'b':
            isGray = !isGray;
            break;
        case '+':
            rate += 0.1;
            break;
        case '-':
            rate -= 0.1;
            break;
        }
    }
    return 0;
}
