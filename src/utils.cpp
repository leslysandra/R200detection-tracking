/* utils.cpp
*/

#include "utils.h"
#include "common.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>

using namespace std;
using namespace cv;

const static int SENSITIVITY_VALUE = 20;
// bounding rectangle of the object (using the center of this as its position)
cv::Rect objectBoundingRectangle = cv::Rect(0,0,0,0);

/* trackUser -- Function used to track color blobs on a RGB image. */
void trackUserByColor(cv::Mat& src, cv::Mat& regmask) {
	Mat frame(src.size(), src.type());                          // set dimensions
	Mat(src).copyTo(frame);					    // copy
        Mat hsv = cv::Mat::zeros(frame.size(), frame.type());       // define container for HSV mat
	cvtColor(frame, hsv, cv::COLOR_BGR2HSV);                   // convert

	// MASK 
	Mat mask;
	// look for green color
	inRange(hsv,cv::Scalar(hMin,sMin,vMin), cv::Scalar(hMax,sMax,vMax), mask);
	Mat erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	Mat dilateElement = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

	// morphological opening (remove small objects from the foreground)
    	erode(mask, mask, erodeElement);
	erode(mask, mask, erodeElement);
	dilate(mask, mask, dilateElement);
	// morphological closing (fill small holes in the foreground)
  	dilate(mask, mask, dilateElement);
  	erode(mask, mask, erodeElement);

	// blur
	blur(mask, mask, Size(SENSITIVITY_VALUE, SENSITIVITY_VALUE));
	// threshold again to obtain binary image from blur output
	threshold(mask, mask, SENSITIVITY_VALUE, 255, THRESH_BINARY);

	imshow("mask",mask);           // exhibit mask.

	// find the contours in the mask and initialize the current (x, y) center of the detected region
	vector<vector<Point> > contours;  // container for the contours
	vector<Vec4i> hierarchy;

	// void findContours(InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset=Point())
	findContours(mask.clone(), contours, hierarchy,
                     CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    	Point2f center(-1000,-1000);                // initialize center in random value
    	missedPlayer = true;                        // flag to track the Player presence

	// if at least one contour was found (the contour correspond to the color blob dettected)
	if (contours.size() > 0) {

		int largest_area=0;               // container for the max area
		int largest_contour_index=0;      // container for the index of the max area found in the countours

		//find the largest contour to compute the minimum enclosing circle and centroid
		for(int i=0; i < contours.size();i++){
			double a = contourArea(Mat(contours[i]),false);  //  Find the area of contour
            		// update to always get the bigger area
           		if(a > largest_area) {
				largest_area = a;
				largest_contour_index = i;
	    		}
		}

		Point2f tempcenter;
  		float radius;
		// cv::minEnclosingCircle(points, center, radius)
		minEnclosingCircle((Mat)contours[largest_contour_index], tempcenter, radius);

		// cv::moments(array, bnaryImage)
		Moments M = moments((Mat)contours[largest_contour_index]);
		center = cv::Point2f(int(M.m10 / M.m00), int(M.m01 / M.m00));

		// if radius has a minimum size to be considered the tracked blob
		if (radius > 15) {
			// draw the circle and centroid on the frame, then update the list of tracked points
			circle(frame, Point(int(tempcenter.x), int(tempcenter.y)), int(radius), Scalar(0, 255, 255), 2);
			circle(frame, center, 5, cv::Scalar(0, 0, 255), -1); // red point in the center
		    	mainCenter = center;            // save the center of the detected circle
		    	missedPlayer = false;           // update the flag for the player presence
		}
	}

	// update the points queue
	pts.push_front(center);

        // looking over the tracked point to reprent movement history (red line)
	for (int i=1; i < (pts.size()-1); i++) {
		// ignore points that are NONE. (NONE = -1000)
		cv::Point2f ptback = pts[i - 1];
		cv::Point2f pt = pts[i];
		if ((ptback.x == -1000) or (pt.x == -1000))  continue;

		// otherwise, compute the thickness of the line and draw the connecting lines
		int thickness = int(sqrt(BUFFER / float(i + 1)) * 2.5);
		line(frame, pts[i - 1], pts[i], cv::Scalar(0, 0, 255), thickness);
	}

	src = frame.clone();   // update the input frame
	regmask = mask.clone(); // return value of mask
}

/*
// another method to detect and search
void searchForMovement(cv::Mat thresholdImage, cv::Mat &cameraFeed) {
	bool objectDetected = false;
	cv::Mat temp;
	thresholdImage.copyTo(temp);
	//vectors for output of findContours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);// retrieves external contours

	//if contours vector is not empty, we have found some objects
	if(contours.size() > 0) objectDetected=true;
	else objectDetected = false;

	if(objectDetected) {
		// the largest contour is found at the end of the contours vector
		// we will simply assume that the biggest contour is the object we are looking for.
		std::vector< std::vector<cv::Point> > largestContourVec;
		largestContourVec.push_back(contours.at(contours.size()-1));
		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.
		objectBoundingRectangle = boundingRect(largestContourVec.at(0));
		int xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
		int ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;

		//update the objects positions by changing the 'theObject' array values
		theObject[0] = xpos;
		theObject[1] = ypos;
	}

	//make some temp x and y variables so we dont have to type out so much
	int x = theObject[0];
	int y = theObject[1];

	//draw some crosshairs around the object
	circle(cameraFeed, cv::Point(x,y), 30, cv::Scalar(0,255,0), 2);
	line(cameraFeed, cv::Point(x,y), cv::Point(x,y-25), cv::Scalar(0,255,0), 1);
	line(cameraFeed, cv::Point(x,y), cv::Point(x,y+25), cv::Scalar(0,255,0), 1);
	line(cameraFeed, cv::Point(x,y), cv::Point(x-25,y), cv::Scalar(0,255,0), 1);
	line(cameraFeed, cv::Point(x,y), cv::Point(x+25,y), cv::Scalar(0,255,0), 1);

	//write the position of the object to the screen
	putText(cameraFeed,"Tracking object at ", cv::Point(x,y), 1, 1, cv::Scalar(255,0,0),2);
}*/
