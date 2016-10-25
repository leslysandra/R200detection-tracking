/* main.cpp
*/

/* librealsense NO_WARNINGS */
#define _CRT_SECURE_NO_WARNINGS
#define BASE_PATH "/home/aerolabio/librealsense/examples/librealsense_feat"

/* std */
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <sstream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <utility>
#include <stdlib.h>
#include <cstdlib>
#include <boost/circular_buffer.hpp>

/* sqrt */
#include <math.h>

/* OpenCV related includes */
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/objdetect.hpp"

/* librealsense libraries */
#include <librealsense/rs.hpp>

/* for timing */
#include <chrono>
#include <thread>

/* local includes */
#include "utils.h"
#include "common.h"

using namespace std;
using namespace cv;

/* HSV space variables for PURPLE blob detection */
int hMin = 120;
int sMin = 148;
int vMin = 55;
int hMax = 256;
int sMax = 256;
int vMax = 130;

// threshold value
int thresh = 100;

// writing into external file the data that will be plotted
ofstream timeStatsFile_;
// contain motion points
map<int, int> distribution;

// variable for blob center tracking
Point2f mainCenter;
// flag for the player presence
bool missedPlayer;

// the object trail size
const int BUFFER = 32;
// The blob location history
boost::circular_buffer<cv::Point2f> pts(BUFFER);

// Whether the running application should shut down
bool protonect_shutdown = false;

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
		line(aux, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(255,255,255));
		circle(aux, Point(x,y), 2, Scalar(255,255,255), -1);
	    }
        }
    }
}

void cleanDist() {
	distribution.clear();
}

void insertInDist(float d) {
	distribution[(int)d]++;
}

void printDist() {
	for(auto p : distribution) {
		cout << p.first << " |";
		for(int i = 0; i < p.second; i++) cout << "·";
		cout << endl;
	}
		cout << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl;
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

	// recording and writing into --> "output.avi"
	int codec = CV_FOURCC('M', 'J', 'P', 'G');
	cv::VideoWriter out = VideoWriter("output.avi", codec, 10.0, Size(640, 480), true);

  // openCV frame containers
  Mat rgbmat, depthmat, depthAndRgb, regframe, aux, output;

  // openCV frame definition for Optical Flow
	Mat gray, flow, sflow, flow1, greenFlowMat;
	UMat prevgray;

	// timing for plotting
	auto startT = std::chrono::system_clock::now();

	// blob detection image panel
  cv::namedWindow("mask", 1);

  // track bar for the blob color detected image
  cv::createTrackbar("hMin", "mask", &hMin, 256);
  cv::createTrackbar("sMin", "mask", &sMin, 256);
  cv::createTrackbar("vMin", "mask", &vMin, 256);
  cv::createTrackbar("hMax", "mask", &hMax, 256);
  cv::createTrackbar("sMax", "mask", &sMax, 256);
  cv::createTrackbar("vMax", "mask", &vMax, 256);

  // FEATURE VARIABLES
  // distance feature
  float meanDistance = 0;

	// capture first 50 frames to allow camera to stabilize
	for (int i = 0; i < 50; ++i) dev->wait_for_frames();

	// save data in csv to plot later
	timeStatsFile_.open("../gnuplot/pos-time.csv");
	// opening with complete path:
	//timeStatsFile_.open("/home/aerolabio/librealsense/examples/librealsense_feat/gnuplot/pos-time.csv");
	timeStatsFile_ << "t, x, y, d" << endl;


	// loop -- DATA ACQUISITION
	while (true) {

		// wait for new frame data
		dev->wait_for_frames();

    // image frame acquisition
		// RGB data acquisition
		uchar *rgb = (uchar *) dev->get_frame_data(rs::stream::color);
		// DEPTH data acquisition
		uchar *depth = (uchar *) dev->get_frame_data(rs::stream::depth);

		// data acquisition into opencv::Mat
		// RGB
		const uint8_t * rgb_frame = reinterpret_cast<const uint8_t *>(rgb);
		cv::Mat rgb_ = cv::Mat(480, 640, CV_8UC3, (void*) rgb_frame);
    // saving RGB data into (mat container)rgbmat
		cvtColor(rgb_, rgbmat, CV_BGR2RGB);

    // DEPTH
		const uint8_t * depth_frame = reinterpret_cast<const uint8_t *>(depth);
		Mat depth16(480, 2*640, CV_8U, (void*) depth_frame);
    Mat depthM(depth16.size().height, depth16.size().width, CV_16UC1);
    depth16.convertTo(depthM, CV_8UC3);
    // min/max distance from the camera
		unsigned short min = 0.5, max = 3.5;
		cv::Mat img0 = Mat::zeros(depthM.size().height, depthM.size().width, CV_8UC1);
		cv::Mat depth_show;
		double scale_ = 255.0 / (max-min);
		depthM.convertTo(img0, CV_8UC1, scale_);

		applyColorMap(depthM, depthmat, cv::COLORMAP_JET); // ColorMap to depthmat

		// OPTICAL FLOW
		Mat frame(rgbmat.size(), rgbmat.type());
		rgbmat.copyTo(frame);
    // readjusting original image to gray scale image
	  cvtColor(frame, gray, COLOR_RGB2GRAY);

		if (!prevgray.empty()) {
			flow = Mat(gray.size(), CV_32FC2);

			// Optical Flow operation
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

			// optical flow points
			sflow = Mat(flow.size(), CV_8UC3);
			aux = Mat::ones(flow.size(), CV_8U);
      // obtain matrix (sflow, aux) with points detected for motion
			drawOptFlowMap(flow, sflow, aux, 4, 1.5, Scalar(0, 255, 0));
			//imshow("sflow", sflow); imshow("flow_aux", aux);
			aux.convertTo(flow1, CV_8U);

			// magnitude,angle
			Mat xy[2];
			split(flow, xy);
			Mat magnitude, angle; //calculate angle and magnitude
			cartToPolar(xy[0], xy[1], magnitude, angle, true);
			double mag_max;
			minMaxLoc(magnitude, 0, &mag_max);
			magnitude.convertTo(magnitude, 0.0, 255.0/mag_max); // magnitude extracted from flow
			//imshow("magnitudeFlow", magnitude);

		} else gray.copyTo(prevgray);

		// method to track the color blob representing the player
		Mat frameToTrack(rgbmat.size(), rgbmat.type());
		rgbmat.copyTo(frameToTrack);
		trackUserByColor(frameToTrack, regframe);

		// flow matrix AND green track
    if (flow1.cols > 0 && flow1.rows > 0) {
			bitwise_and(flow1, regframe, greenFlowMat);
			circle(greenFlowMat, mainCenter, 5, Scalar(255, 255, 255), -1);

			// show matrix greenBlobMat && flowMat
			imshow("greenFlowMat", greenFlowMat);

			// plotting
			vector<Point2f> movingGreenPoints;	// movingGreenPoints vector
			vector<float> movingGreenDepthPoints;   // movingGreenDepthPoints vector

			int step = 16;
			float mean_x = 0;
			float mean_y = 0;
			float mean_d = 0;
			int movingGreenDepthPointsCount = 0;

			for(int y = 0; y < greenFlowMat.rows; y += step)
				for(int x = 0; x < greenFlowMat.cols; x += step)
					if(greenFlowMat.at<unsigned short int>(y, x)) {
						movingGreenPoints.push_back(Point2f(x, y));
						float d = (float) depth16.at<unsigned short int>(y, x);
						if(d>10 && d<5000) { // [mm]
							mean_d += d;
							movingGreenDepthPointsCount++;
							timeStatsFile_ << mean_d << endl;
              // save float point "d" extracted from depth matrix (to get the distance)
							insertInDist(d);
              // TODO verify "d" and extract distance from the camera to the object
						}
					}

      // if vector with moving points exists
			if (movingGreenPoints.size()) {
        // average values of moving points
				mean_x = mean(movingGreenPoints)[0];
				mean_y = mean(movingGreenPoints)[1];

        // TODO review if this value is reliable
				// mean distance
				if(movingGreenDepthPointsCount) mean_d /= movingGreenDepthPointsCount;

				// normalization (so x,y range will be -1 to 1)
				float norm_mean_x = (2*mean_x)/greenFlowMat.rows - 1;
				float norm_mean_y = (2*mean_y)/greenFlowMat.cols - 1;

				// get the current time
				auto currentT = std::chrono::system_clock::now();
        // get the time duration for last operation
				auto dur = currentT - startT;
        // time in seconds
				typedef std::chrono::duration<float> float_seconds;
				auto secs = std::chrono::duration_cast<float_seconds>(dur);

        // data that will be saved to csv file to be plotted
				timeStatsFile_ << secs.count() << ", "
                        << norm_mean_x << ", "
                        << norm_mean_y << ", "
                        << mean_d << endl;

        // distribution contains information about moving points in depth matrix
				cout << "distribution" << endl;
				printDist();
				cleanDist();

        // TODO method to use the “distance value, obtained” to plot it or send to bag file in order to communicate with other systems

        // display matrix when object was detected by color and moving points
				imshow("trackingColor-MatRGB",frameToTrack);
				//imshow("depth16", depth16);
				//imshow("depthmat", depthmat);

			}

    }

		// calling the plot
		//system("/home/aerolabio/librealsense/examples/librealsense_feat/gnuplot/gp.sh");

		// save recorded video written in --> output.avi
		out.write(frameToTrack);

		// waitKey needed for showing the plots with cv::imshow
		char key = cv::waitKey(10);
		if(key == 27) {
			//cleaning up
 			cvDestroyAllWindows();
 			// closing csv
 			timeStatsFile_.close();
 			break;
		}
		// update prev frame
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
