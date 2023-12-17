#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;


// set up configuration

float confidenceThreshold = 0.6;

string modelConfiguration = "/home/ahmed/testOpenCV/data/deploy.prototxt.txt";
string modelBinary = "/home/ahmed/testOpenCV/data/res10_300x300_ssd_iter_140000.caffemodel";


const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanValue(104.0, 177.0, 123.0);


int main(){

    // init dnn

    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);

    if (net.empty()) {
        cerr << "Can't load network by using the following files: " <<
        endl;
        cerr << "prototxt: " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models are available here:" << endl;

        exit(-1);
    }    // end if


    VideoCapture cap;
    cap.open(0);

    if(!cap.isOpened()){
        cout << "can't open camera" << endl;
     return -1;
    }
    while(true){
        Mat frame;
        double timer = (double)getTickCount();
        cap >> frame;

        if (frame.empty()){
            waitKey();
            break;
        }   // end if
        Mat inputBlob = blobFromImage(frame, inScaleFactor,
        Size(inWidth, inHeight), meanValue, false, false);

        net.setInput(inputBlob, "data"); //set the network input
        //! [Make forward pass]
        Mat detection = net.forward("detection_out"); //compute output
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F,
        detection.ptr<float>());

        for(int i = 0; i < detectionMat.rows; i++)
        {
        float confidence = detectionMat.at<float>(i, 2);
        if(confidence > confidenceThreshold)
        {
        int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3)
        * frame.cols);
        int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4)
        * frame.rows);
        int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
        int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

        Point p1(xRightTop, yRightTop);
        Point p2(xLeftBottom, yLeftBottom);


        rectangle(frame, p1, p2, Scalar(0, 255, 0));
        String label = "Face: " + to_string(confidence);
        int baseLine = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_DUPLEX, 0.5, 1, &baseLine);
        rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom -
        labelSize.height),
        Size(labelSize.width, labelSize.height + baseLine)),
        Scalar(255, 255, 255), FILLED);

        float fps = getTickFrequency() / ((double)getTickCount() - timer);
           // Display FPS on frame
        putText(frame, "FPS : " + to_string(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        putText(frame, label, Point(xLeftBottom, yLeftBottom),
        FONT_HERSHEY_DUPLEX, 0.5, Scalar(0,0,0));
    }  // end if



    } // end for

    imshow ( "faceDetection", frame );

    if ( cv::waitKey (1) >= 0 ){

        break;

    }

    }// end while


    return 0;
}

