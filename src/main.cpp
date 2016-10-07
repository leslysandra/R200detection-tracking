/* main.cpp
*/

/* librealsense NO_WARNINGS */
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <utility>
#include <cstdlib>
#include <boost/circular_buffer.hpp>
#include <math.h>       /* sqrt */

/* OpenCV related includes */
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/objdetect.hpp"
/* ... */

/* librealsense libraries */
#include <librealsense/rs.hpp>
/* ... */

/* Local includes */
#include "utils.h"
#include "common.h"
/* ... */

/* HSV space variables for GREEN blob detection */
int hMin = 40;
int sMin = 100;
int vMin = 80;
int hMax = 70;
int sMax = 255;
int vMax = 255;
/* ... */

int thresh = 100;


using namespace std;
using namespace cv;


cv::Point2f mainCenter;         // variable for blob center tracking
bool missedPlayer;              // a flag for the player presence.

const int BUFFER = 32;          			// the object trail size
boost::circular_buffer<cv::Point2f> pts(BUFFER);    	// The blob location history

bool protonect_shutdown = false;        // Whether the running application should shut down.

/* Interruption handler function. In this case, set the variable that controls the
 frame aquisition, breaking the loop, cleaning variables and exit the program elegantly*/
void sigint_handler(int s){
    protonect_shutdown = true;
}

/* Creates the optFlow map*/
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, Mat& aux, int step, double, const Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step) {
        for(int x = 0; x < cflowmap.cols; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x);

            // drawing the lines of motion
	    line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(255,0,0));
            circle(cflowmap, Point(x,y), 2, color, -1);

	    if( (fabs(fxy.x)>5) && (fabs(fxy.y)>5) ) {		
		//line(aux, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(255,0,0));
		circle(aux, Point(x,y), 2, Scalar(255,255,255), -1); 
	    }
        }
    }
}

/* Creates the blob/optFlow map*/
static void boblification(const Mat& flow, Mat& flowAndColor, Mat& result, int sX, int sY) {
    	long int pixels = 0; // pixels counter
	vector<vector<int> > reach;	       // binary mask for the segmentation.
	for (int i = 0; i < flow.rows; i++){	// values in zero, no pixel is assigned to the segmentation.
		reach.push_back(vector<int>(flow.cols));
	}

	// Define the queue. NOTE: it is a BFS based algorithm.
	std::queue< std::pair<int,int> > seg_queue;

	cout << "sX: " << sX << " - sY: " << sY << endl;

	// verify the depth value of the seed position.
	//float &in_pxl_pos = flow.at<float>(sY,sX);
	//cout << "center value: " << in_pxl_pos << endl;
}

/* main */
int main(int argc, char** argv)
try {
    	// Turn on logging.
	rs::log_to_console(rs::log_severity::warn);
	std::cout << "Starting..." << std::endl;

	// realsense context
	rs::context ctx;
        cout << "There are " << ctx.get_device_count() << " connected RealSense devices." << endl << endl;

	// exit if not device is already connected
        if (ctx.get_device_count() == 0) return EXIT_FAILURE;

	// rs defining device to be used
	rs::device * dev = ctx.get_device(0);

        cout << "Using device..."<< endl <<"   name: " << dev->get_name() << endl << "   serial number: "
	<< dev->get_serial() << endl << "   firmware version: " << dev->get_firmware_version() 
	<< endl << endl;

	// configure DEPTH to run at 60 frames per second
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 60);
        // configure RGB to run at 60 frames per second
	dev->enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 60);

	// settings to improve the DEPTH image
	rs::apply_depth_control_preset(dev, 3);

        cout << "Device is warming up... " << endl << endl;

	// start the device
	dev->start();

	// recording
	//int codec = CV_FOURCC('M', 'J', 'P', 'G');
	//cv::VideoWriter out = VideoWriter("output.avi", codec, 60.0, Size(640, 480), true);

    	// openCV frame containers
    	cv::Mat rgbmat, depthmat, depthAndRgb, regframe, aux, output;

    	// openCV frame definition for Optical Flow
	Mat gray, flow, sflow, flow1;
	UMat prevgray;

	// blob detection image panel
    	cv::namedWindow("mask", 1);

    	cv::createTrackbar("hMin", "mask", &hMin, 256);
    	cv::createTrackbar("sMin", "mask", &sMin, 256);
    	cv::createTrackbar("vMin", "mask", &vMin, 256);
    	cv::createTrackbar("hMax", "mask", &hMax, 256);
    	cv::createTrackbar("sMax", "mask", &sMax, 256);
    	cv::createTrackbar("vMax", "mask", &vMax, 256);

    	/* FEATURE VARIABLES */
    	float meanDistance = 0;         // distance feature

	// capture first 50 frames to allow camera to stabilize
	for (int i = 0; i < 50; ++i) dev->wait_for_frames();

	// loop -- DATA ACQUISITION
	while (true) {

		// wait for new frame data
		dev->wait_for_frames();

		// RGB data acquisition
		uchar *rgb = (uchar *) dev->get_frame_data(rs::stream::color);
		// DEPTH data acquisition
		uchar *depth = (uchar *) dev->get_frame_data(rs::stream::depth);

		// data acquisition into opencv::Mat
		// RGB
		const uint8_t * rgb_frame = reinterpret_cast<const uint8_t *>(rgb);
		cv::Mat rgb_ = cv::Mat(480, 640, CV_8UC3, (void*) rgb_frame);
		cvtColor(rgb_, rgbmat, CV_BGR2RGB); // saving data into cv::mat container rgbmat

                // DEPTH
		const uint8_t * depth_frame = reinterpret_cast<const uint8_t *>(depth);
		cv::Mat depth16(480, 640, CV_16UC1, (void*) depth_frame);
                cv::Mat depthM(depth16.size().height, depth16.size().width, CV_16UC1);
                depth16.convertTo(depthM, CV_8UC3);
		// min/max distance from the camera
		unsigned short min = 0.5, max = 3.5;
		cv::Mat img0 = cv::Mat::zeros(depthM.size().height, depthM.size().width, CV_8UC1);
		cv::Mat depth_show;
		double scale_ = 255.0 / (max-min);

		depthM.convertTo(img0, CV_8UC1, scale_);
		cv::applyColorMap(img0, depthmat, cv::COLORMAP_JET); // ColorMap to depth matrix

		// TEST OPTICAL FLOW
		Mat frame(rgbmat.size(), rgbmat.type());
		rgbmat.copyTo(frame);
	   	cvtColor(frame, gray, COLOR_RGB2GRAY);

		if (!prevgray.empty()) {
			flow = Mat(gray.size(), CV_32FC2);

			// Optical Flow
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

			// optical flow points
			sflow = Mat(flow.size(), CV_8UC3);
			aux = Mat::ones(flow.size(), CV_8U);
			drawOptFlowMap(flow, sflow, aux, 16, 1.5, Scalar(0, 255, 0));
			//imshow("sflow", sflow); //imshow("flow_aux", aux);

			//cvtColor(sflow, flow1, CV_BGR2GRAY);
			aux.convertTo(flow1, CV_8U);

			// magnitude,angle
			cv::Mat xy[2];
			split(flow, xy);

			//calculate angle and magnitude
			Mat magnitude, angle;
			cartToPolar(xy[0], xy[1], magnitude, angle, true);
			double mag_max;
			minMaxLoc(magnitude, 0, &mag_max);
			magnitude.convertTo(magnitude, 0.0, 255.0/mag_max);
			//imshow("magnitudeFlow", magnitude);

		} else gray.copyTo(prevgray);

		// method to track the color blob representing the human
		Mat frameToTrack(rgbmat.size(), rgbmat.type());
		rgbmat.copyTo(frameToTrack);
		trackUser(frameToTrack, regframe);

		// flow matrix AND green track
     		if (flow1.cols > 0 && flow1.rows > 0) {
     			Mat res;
     			addWeighted( flow1, 0.5, regframe, 0.5, 0.0, res);
			//bitwise_and(flow1, regframe, res, regframe);
			circle(res, mainCenter, 5, cv::Scalar(255, 255, 255), -1);
			imshow("res", res);
			// boblificatonMethod(input: flow1, res, output)
			//boblification(flow1, res, output, mainCenter.x, mainCenter.y);
     		}


		/*// detecting borders
		cv::Mat clonemask = regframe.clone();
		cv::Mat canny_output;
  		std::vector<std::vector<cv::Point> > contours_;
  		std::vector<cv::Vec4i> hierarchy_;
  		// detect edges using canny
  		Canny(clonemask, canny_output, thresh, thresh*2, 3);
  		// findContours
 		cv::findContours(canny_output, contours_, hierarchy_, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  		// draw contours
  		cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3 );
 		for(int i = 0; i< contours_.size(); i++ ) {
			// only countours     			
			drawContours(drawing, contours_, i, cv::Scalar(255,0,0), 2, 8, hierarchy_, 0, cv::Point());
     		}*/

		// Update/show images
		//imshow("original RGB",rgbmat);
		imshow("afterTRACK",frameToTrack);
		//imshow("regframe",regframe);
       	 	//imshow("depth", depthmat);
  		//imshow( "Contours", drawing);

		// save recorded video
		//out.write(rgbmat);

		// waitKey needed for showing the plots with cv::imshow
		char key = cv::waitKey(10);
		if(key == 27) {
			//cleaning up
 			cvDestroyAllWindows();
 			break;
		}

		gray.copyTo(prevgray);		
	}
    std::cout << "Shutting down!" << std::endl;
    return 0;
} catch (const rs::error & e) {
	// method calls against librealsense objects may throw exceptions of type rs::error
	cout << "rs::error was thrown when calling " << e.get_failed_function().c_str()
	<< "-" << e.get_failed_args().c_str() << "-:     " << e.what() << endl;
	return EXIT_FAILURE;
}
