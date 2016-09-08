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

/* HSV space variables for blob detection */
int hMin = 30;
int sMin = 70;
int vMin = 64;
int hMax = 70;
int sMax = 255;
int vMax = 255;
/* ... */

/* HSV space variables for RED blob detection */
int hMinR = 0;
int sMinR = 200;
int vMinR = 0;
int hMaxR = 19;
int sMaxR = 255;
int vMaxR = 255;


int thresh = 100;
int max_thresh = 255;


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

        // camera
	int cam = 0;

	// rs defining device to be used
	rs::device * dev = ctx.get_device(cam);

        cout << "Using device..."<< endl <<"   name: " << dev->get_name() << endl << "   serial number: "
	<< dev->get_serial() << endl << "   firmware version: " << dev->get_firmware_version() 
	<< endl << endl;

	// configure DEPTH to run at 60 frames per second
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 60);
        // configure RGB to run at 60 frames per second
	dev->enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 60);

	// settings to improve the DEPTH image
	rs::apply_depth_control_preset(dev, 3);
	//rs::option::r200_depth_control_neighbor_threshold;

        cout << "Device is warming up... " << endl << endl;

	// start the device
	dev->start();

	// recording image
	Size size_ = Size(640, 480);
	int codec = CV_FOURCC('M', 'J', 'P', 'G');
	cv::VideoWriter out = VideoWriter("output.avi", codec, 60.0, size_, true);

    	// OpenCV frame definition. They are the containers from which frames are writen to and exihibit.
    	cv::Mat rgbmat, depthmat, depthAndRgb, regframe;
    	int count = 0;

	/* Create the blob detection image panel together with the
        sliders for run time adjustments. */
    	cv::namedWindow("mask", 1);

    	cv::createTrackbar("hMin", "mask", &hMin, 256);
    	cv::createTrackbar("sMin", "mask", &sMin, 256);
    	cv::createTrackbar("vMin", "mask", &vMin, 256);
    	cv::createTrackbar("hMax", "mask", &hMax, 256);
    	cv::createTrackbar("sMax", "mask", &sMax, 256);
    	cv::createTrackbar("vMax", "mask", &vMax, 256);
    	/* ---- */

    	/* FEATURE VARIABLES */
    	float meanDistance = 0;         // distance feature
    	float ci = 0;                   // contraction index

	bool acquire = true;		// control WHILE loop

	// capture first 50 frames to allow camera to stabilize
	for (int i = 0; i < 50; ++i) dev->wait_for_frames();

	// loop -- DATA ACQUISITION
	while (acquire) {

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
		cv::applyColorMap(img0, depthmat, cv::COLORMAP_JET); // saving data into cv::mat container depthmat

		// Search for the color blob representing the human
		trackUser(rgbmat, regframe);

		// detecting borders
		cv::Mat clonemask = regframe.clone();
		cv::Mat canny_output;
  		std::vector<std::vector<cv::Point> > contours_;
  		std::vector<cv::Vec4i> hierarchy_;

  		// Detect edges using canny
		// void Canny(input, output_edges, threshold1, threshold2, int apertureSize=3, bool L2gradient=false)
  		Canny(clonemask, canny_output, thresh, thresh*2, 3);
  		// findContours
 		cv::findContours(canny_output, contours_, hierarchy_, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  		// Draw contours
  		cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
 		for( int i = 0; i< contours_.size(); i++ ) {
       			drawContours(drawing, contours_, i, cv::Scalar(255,0,0), 2, 8, hierarchy_, 0, cv::Point());
     		}

  		// Show in a window
  		cv::imshow( "Contours", drawing);

		// obtain the image ROI
		cv::Mat undistortedFrame = cv::Mat(regframe.size(), CV_8UC1);
		cv::Mat segmat = Mat::zeros(regframe.size(), undistortedFrame.type());
		cv::Mat roiSegment = Mat::zeros(regframe.size(), undistortedFrame.type());

		//THIS LOOP COMPUTES A CIRCLE BASED ON THE mainCenter VARIABLE COMPUTED BY THE trackUser METHOD.
		if ((mainCenter.x != -1000) && (mainCenter.x != 0)){
			int radius = 5;

			cout << "mainCenter.x: " << mainCenter.x << " - mainCenter.y: " << mainCenter.y 
	<< " radius: " << radius << endl;
			//cout << "rgbmat(x,y): " << rgbmat.at<float>(mainCenter.y, mainCenter.x) << endl;
			//cout << "depthmat(x,y): " << depthmat.at<float>(mainCenter.y, mainCenter.x) << endl << endl;

			//get the Rect containing the circle:
		    	cv::Rect r(mainCenter.x-radius, mainCenter.y-radius, radius*2,radius*2);

			// region of interest
		    	cv::Mat roi(regframe, r);

		    	// make a black mask, same size:
		    	cv::Mat maskROI(roi.size(), roi.type(), cv::Scalar::all(0));

		    	// with a white, filled circle in it:
		    	cv::circle(maskROI, cv::Point(radius,radius), radius, cv::Scalar::all(255), -1);

		    	// combine roi & mask:
		    	cv::Mat roiArea = roi & maskROI;

		    	cv:Scalar m = cv::mean(roi);        // compute mean value of the region of interest (mm).
		    	meanDistance = m[0] / 1000.0f;     // compute distance (in meters)

			cout << "distance: " << meanDistance << endl;

		    	/* perform segmentation in order to get the contraction index feature.
			The result will be saved in ci variable*/
		    	// void segmentDepth(cv::Mat& input, cv::Mat& dst, cv::Mat& roiSeg, int sX, int sY, int threshold)
		    	segmentDepth(regframe, segmat, roiSegment, mainCenter.x, mainCenter.y, ci, 300);

		}


		/* A LOOP TO PRINT THE DISTANCE AND HISTORY TRACE IN THE DEPTH FRAME. IT JUST SERVES AS A
         	VISUAL AID OF WHAT IS GOING ON.*/
    		for (int i=1; i < (pts.size()-1); i++){
	    		// if either of the tracked points are None, ignore
	    		// them

	    		cv::Point2f ptback = pts[i - 1];
	    		cv::Point2f pt = pts[i];
	    		if ((ptback.x == -1000) or (pt.x == -1000)){
		        continue;
	    		}

	    		// otherwise, compute the thickness of the line and
	    		// draw the connecting lines
	    		int thickness = int(sqrt(BUFFER / float(i + 1)) * 2.5);
	    		line(regframe, pts[i - 1], pts[i], cv::Scalar(0, 0, 255), thickness);
		    // Write distance
		    cv::putText(regframe,
		    std::to_string(meanDistance),
		    cv::Point((512/2)-60,85), // Coordinates
		    	cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
		    	1.0, // Scale. 2.0 = 2x bigger
		    	cv::Scalar(255,255,255), // Color
		    	1 // Thickness
		    ); // Anti-alias
	    	}


		/* Update/show images */
		cv::imshow("afterTRACK",rgbmat);
       	 	//cv::imshow("depth", depthmat);
        	//cv::imshow("undistortedFrame", undistortedFrame);
		cv::imshow("segmat", segmat);
		//cv::imshow("roiSegment", roiSegment);

		out.write(rgbmat);

		// waitKey needed for showing the plots with cv::imshow
		char key = cv::waitKey(1);
		if(key == 27) {
			acquire = false;

			//cleaning up
 			cvDestroyAllWindows(); 
		}
		
	}

	// TODO close session with DEV
    //dev->stop();
    //dev->close();
    std::cout << "Shutting down!" << std::endl;
    return 0;
} catch (const rs::error & e) {
	// method calls against librealsense objects may throw exceptions of type rs::error
	cout << "rs::error was thrown when calling " << e.get_failed_function().c_str()
	<< "-" << e.get_failed_args().c_str() << "-:     " << e.what() << endl;
	return EXIT_FAILURE;
}
