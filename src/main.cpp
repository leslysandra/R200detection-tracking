/* main.cpp
*/

/* librealsense NO_WARNINGS */
#define _CRT_SECURE_NO_WARNINGS

/* std */
#include <iostream>
#include <fstream>
#include <string>
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

ofstream timeStatsFile_;

Point2f mainCenter;         // variable for blob center tracking
bool missedPlayer;          // flag for the player presence.

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
		line(aux, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(255,255,255));
		circle(aux, Point(x,y), 2, Scalar(255,255,255), -1); 
	    }
        }
    }
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
    	Mat rgbmat, depthmat, depthAndRgb, regframe, aux, output;

    	// openCV frame definition for Optical Flow
	Mat gray, flow, sflow, flow1, greenFlowMat;
	UMat prevgray;

	// timing for plotting
	auto startT = std::chrono::system_clock::now();

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

	// save data in csv to plot later
	timeStatsFile_.open("/home/aerolabio/LeslyTracker/pos-time.csv");
	timeStatsFile_ << "t, x, y" << endl;


	// loop -- DATA ACQUISITION
	while (true) {

		// wait for new frame data
		dev->wait_for_frames();

		// RGB data acquisititimeon
		uchar *rgb = (uchar *) dev->get_frame_data(rs::stream::color);
		// DEPTH data acquisition
		uchar *depth = (uchar *) dev->get_frame_data(rs::stream::depth);

		// data acquisition into opencv::Mat
		// RGB
		const uint8_t * rgb_frame = reinterpret_cast<const uint8_t *>(rgb);
		cv::Mat rgb_ = cv::Mat(480, 640, CV_8UC3, (void*) rgb_frame);
		cvtColor(rgb_, rgbmat, CV_BGR2RGB); // saving RGB data into (mat container)rgbmat

                // DEPTH
		const uint8_t * depth_frame = reinterpret_cast<const uint8_t *>(depth);
		cv::Mat depth16(480, 640, CV_16UC1, (void*) depth_frame);
                cv::Mat depthM(depth16.size().height, depth16.size().width, CV_16UC1);
                depth16.convertTo(depthM, CV_8UC3);
		// min/max distance from the camera
		unsigned short min = 0.5, max = 3.5;
		cv::Mat img0 = Mat::zeros(depthM.size().height, depthM.size().width, CV_8UC1);
		cv::Mat depth_show;
		double scale_ = 255.0 / (max-min);
		depthM.convertTo(img0, CV_8UC1, scale_);
		cv::applyColorMap(img0, depthmat, cv::COLORMAP_JET); // ColorMap to depthmat

		// OPTICAL FLOW
		Mat frame(rgbmat.size(), rgbmat.type());
		rgbmat.copyTo(frame);
	   	cvtColor(frame, gray, COLOR_RGB2GRAY);

		if (!prevgray.empty()) {
			flow = Mat(gray.size(), CV_32FC2);

			// Optical Flow operation
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

			// optical flow points
			sflow = Mat(flow.size(), CV_8UC3);
			aux = Mat::ones(flow.size(), CV_8U);
			drawOptFlowMap(flow, sflow, aux, 16, 1.5, Scalar(0, 255, 0));
			//imshow("sflow", sflow); imshow("flow_aux", aux);

			aux.convertTo(flow1, CV_8U);

			// magnitude,angle
			Mat xy[2];
			split(flow, xy);
			Mat magnitude, angle; //calculate angle and magnitude
			cartToPolar(xy[0], xy[1], magnitude, angle, true);
			double mag_max;
			minMaxLoc(magnitude, 0, &mag_max);
			magnitude.convertTo(magnitude, 0.0, 255.0/mag_max);
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
			vector<Point2f> movingGreenPoints;	// movingPoints vector
			int step = 16;
			float mean_x = 0;
			float mean_y = 0;

			for(int y = 0; y < greenFlowMat.rows; y += step)
				for(int x = 0; x < greenFlowMat.cols; x += step)
					if(greenFlowMat.at<unsigned short int>(y, x))
						movingGreenPoints.push_back(Point2f(x, y));

			if (movingGreenPoints.size()) {
				mean_x = mean(movingGreenPoints)[0];
				mean_y = mean(movingGreenPoints)[1];

				// normalization
				float norm_mean_x = (2*mean_x)/greenFlowMat.rows - 1;
				float norm_mean_y = (2*mean_y)/greenFlowMat.cols - 1;

				// get the current time
				auto currentT = std::chrono::system_clock::now();
				auto dur = currentT - startT;

				typedef std::chrono::duration<float> float_seconds;
				auto secs = std::chrono::duration_cast<float_seconds>(dur);

				timeStatsFile_  << secs.count() << ", " 
				<< norm_mean_x << ", " << norm_mean_y 		
				<< endl;
			}
			
     		}

		// calling the plot
		//system("/home/aerolabio/Desktop/gnuplot_test/gp.sh");

		// Update/show images
		imshow("afterTRACK",frameToTrack);
       	 	//imshow("depth", depthmat);
       	 	cout << "depth.mainCenter: "<< depthmat.at<unsigned short int>(mainCenter.y, mainCenter.x) << endl;
       	 	cout << "mainCenter.y: "<< mainCenter.y << " - mainCenter.x" << mainCenter.x << endl;

		// save recorded video
		//out.write(rgbmat);

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


/* // Creates the blob/optFlow map
static void boblification(const Mat& flow, Mat& flowAndColor, Mat& result, int sX, int sY) {
	int ci;
	result = Mat::zeros(flow.size(), CV_8UC1);
    	long int pixels = 0; // pixels counter
	vector<vector<int> > reach;	       // binary mask for the segmentation.
	for (int i = 0; i < flow.rows; i++){	// values in zero, no pixel is assigned to the segmentation.
		reach.push_back(vector<int>(flow.cols));
	}

	// Define the queue. NOTE: it is a BFS based algorithm.
	std::queue< std::pair<int,int> > seg_queue;

	if (flow.at<float>(sY,sX) != 0) {
		result.at<int>(sY,sX) = 255;

		// Mark the seed as 1, for the segmentation mask.
	    	reach[sY][sX] = 1;
		pixels++;

		// init the queue with seed.
        	seg_queue.push(std::make_pair(sY,sX));
        	while(!seg_queue.empty()) {
        		// pop values
        		std::pair<int,int> s = seg_queue.front();
	    		int x = s.second;
	    		int y = s.first;

			if (x > 0 && y > 0) {
				// lkdflkdj
				seg_queue.pop();

			    	// Right pixel
		    		if((x+1 < flowAndColor.cols) && (!reach[y][x + 1]) &&
			       		(flow.at<float>(y,x+1)!=0)) {
					reach[y][x+1] = true;
					seg_queue.push(std::make_pair(y, x+1));
					// TODO check
			    		//float &pixel = result.at<float>(y,x+1);
					//pixel = 255;
			    		pixels++;;

		    		}

		    		//Below Pixel
		    		if((y+1 < flowAndColor.rows) && (!reach[y+1][x]) &&
					(flow.at<float>(y+1,x)!=0)) {
		    			reach[y+1][x] = true;
		    			seg_queue.push(std::make_pair(y+1,x));
		    			//result.at<int>(y+1,x) = 255;
		    			pixels++;;
		    		}

		    		//Left Pixel
		    		if((x-1 >= 0) && (!reach[y][x-1]) &&
					(flow.at<float>(y,x-1)!=0)) {
		    			reach[y][x-1] = true;
		    			seg_queue.push(std::make_pair(y,x-1));
		    			//result.at<int>(y,x-1) = 255;
		    			pixels++;;
		    		}

		    		//Above Pixel
		    		if((y-1 >= 0) && (!reach[y-1][x]) &&
					(flow.at<float>(y-1,x) !=0)) {
		    			reach[y-1][x] = true;
		    			seg_queue.push(std::make_pair(y-1,x));
		    			//result.at<int>(y-1,x) = 255;
		    			pixels++;;
		    		}

		    		//Bottom Right Pixel
		    		if((x+1 < flowAndColor.cols) && (y+1 < flowAndColor.rows) && (!reach[y+1][x+1]) &&
					(flow.at<float>(y+1,x+1)!=0)) {
		    			//reach[y+1][x+1] = true;
		    			seg_queue.push(std::make_pair(y+1,x+1));
		    			//result.at<int>(y+1,x+1) = 255;
		    			pixels++;
		    		}

		    		//Upper Right Pixel
		    		if((x+1 < flowAndColor.cols) && (y-1 >= 0) && (!reach[y-1][x+1]) &&
					(flow.at<float>(y-1,x+1)!=0)) {
		    			reach[y-1][x+1] = true;
		    			seg_queue.push(std::make_pair(y-1,x+1));
		    			//result.at<int>(y-1,x+1) = 255;
		    			pixels++;
		    		}

		    		//Bottom Left Pixel
		    		if((x-1 >= 0) && (y + 1 < flowAndColor.rows) && (!reach[y+1][x-1]) &&
					(flow.at<float>(y+1,x-1)!=0)) {
		    			reach[y+1][x-1] = true;
		    			seg_queue.push(std::make_pair(y+1,x-1));
		    			//result.at<int>(y+1,x-1) = 255;
		    			pixels++;
		    		}

		    		//Upper left Pixel
		    		if((x-1 >= 0) && (y-1 >= 0) && (!reach[y-1][x-1]) &&
					(flow.at<float>(y-1,x-1)!=0)) {
		    			reach[y-1][x-1] = true;
		    			seg_queue.push(std::make_pair(y-1,x-1));
		    			//result.at<int>(y-1,x-1) = 255;
					pixels++;
		    		}
			}
        	}

		// finding countours for the blob
		vector<vector<Point>> contours;
	    	vector<Vec4i> hierarchy;

		Mat bwImage(flowAndColor.size(),CV_8UC1);
		result.convertTo(bwImage,CV_8U,255.0/(255-0));
	    	cv::findContours(bwImage, contours, hierarchy,
		             CV_RETR_EXTERNAL,
		             CV_CHAIN_APPROX_SIMPLE);

		// continue if at least one countour was found
	    	if (contours.size() > 0) {
		    	Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
		    	Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(8,8));
		    	erode(result, result, erodeElement);
		    	dilate(result, result, dilateElement);

		    	int largest_area=0;
	    	  	int largest_contour_index=0;

	    		// find the largest contour in the mask to compute the minimum enclosing circle
	    		for(int i=0; i < contours.size(); i++) {
	    			// iterate through each contour.
		        	double a = contourArea(Mat(contours[i]),false);  //  Find the area of contour

	    			if( a > largest_area) {
	    				largest_area=a;
	    				largest_contour_index=i;   //Store the index of largest contour
	    			}
	    		}

			// minimum bounding box for the detected blob
			Rect rect = boundingRect(contours[largest_contour_index]);
			Point pt1, pt2;
			pt1.x = rect.x;
			pt1.y = rect.y;
			pt2.x = rect.x + rect.width;
			pt2.y = rect.y + rect.height;

		   	// compute CI
			Mat roiSeg = cv::Mat(result, rect);
			int roiSegarea = roiSeg.total();
			ci = float(roiSegarea-pixels)/float(roiSegarea);

			// compute "middle point"
		    	Mat test = cv::Mat(result,rect);
		    	int begin = 0, loop_counter = 0, row = 10;
		    	bool sign = false;

		    	for (int j=0;j < test.cols;j++) {                 //Basically cost O(n)
				if (test.at<float>(row,j) == 255){
				    if(!sign) {
				        begin = j;
				        sign = !sign;
				    }
				    loop_counter++;
				} else if(sign && test.at<float>(row,j) == 0){
				    break;
				}
		    	}

		    	int mid = std::ceil(loop_counter/2);
		    	int topPoint = begin + (mid-1);

		   	// apply color to frame
		   	Mat segmentedColorFrame, segmentedTarget;
		    	vector<Mat> tempcolorFrame(3);
		    	Mat black = Mat::zeros(result.size(), result.type());
		    	tempcolorFrame.at(0) = black; //for blue channel
		    	tempcolorFrame.at(1) = result;   //for green channel
		    	tempcolorFrame.at(2) = black;  //for red channel

		    	merge(tempcolorFrame, segmentedColorFrame);
			// Draws the rect in the segmentedColorFrame image
		     	circle(segmentedColorFrame, Point(sX,sY),5, Scalar(0,0,255),CV_FILLED, 8,0);
		    	segmentedTarget = cv::Mat(segmentedColorFrame,rect);
		     	circle(segmentedTarget, Point(topPoint,10),5, Scalar(0,0,255),CV_FILLED, 8,0);

			rectangle(segmentedColorFrame, pt1, pt2, cv::Scalar(255,255,255), 2,8,0);

			// putting CI into the frame
		   	putText(segmentedColorFrame, to_string(ci),
		    		Point(rect.x,rect.y-5), // Coordinates
		    		FONT_HERSHEY_COMPLEX_SMALL, // Font
		   	 	0.9, // Scale. 2.0 = 2x bigger
		    		Scalar(255,255,255), // Color
		    		1 // Thickness
		    		); // Anti-alias
		}
	}
}*/
