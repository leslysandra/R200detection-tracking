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

const static int SENSITIVITY_VALUE = 20;
// just one object to search for and keep track of its position
int theObject[2] = {0,0};
// bounding rectangle of the object (using the center of this as its position)
cv::Rect objectBoundingRectangle = cv::Rect(0,0,0,0);

/* trackUser -- Function used to track color blobs on a RGB image. */
void trackUser(cv::Mat& src, cv::Mat& regmask) {
	/* resize the frame and convert it to the HSV color space... */
	cv::Mat frame(src.size(), src.type());                          // set dimensions
	cv::Mat(src).copyTo(frame);					// copy
        cv::Mat hsv = cv::Mat::zeros(frame.size(), frame.type());       // define container for HSV mat
	cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);                   // convert

    	/* construct a mask for the color (default color to "green"), then perform
	a series of dilations and erosions to remove small blobs */
	cv::Mat mask;
	cv::inRange(hsv,cv::Scalar(hMin,sMin,vMin), cv::Scalar(hMax,sMax,vMax), mask);
	//cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
	//cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(8,8));
	cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

    	// perform erode and dilate operations
	// morphological opening (remove small objects from the foreground)
    	cv::erode(mask, mask, erodeElement);
	cv::erode(mask, mask, erodeElement);
	cv::dilate(mask, mask, dilateElement);
	// morphological closing (fill small holes in the foreground)
  	cv::dilate(mask, mask, dilateElement);
  	cv::erode(mask, mask, erodeElement);

	// blur
	cv::blur(mask, mask, cv::Size(SENSITIVITY_VALUE, SENSITIVITY_VALUE));
	// threshold again to obtain binary image from blur output
	cv::threshold(mask, mask, SENSITIVITY_VALUE, 255, cv::THRESH_BINARY);

	cv::imshow("mask",mask);           // exhibit mask.


	/* // red mask for detecting robot
	cv::Mat redmask;
	cv::inRange(hsv,cv::Scalar(0,200,0), cv::Scalar(19,255,255), redmask); // detect RED
	// morphological opening (remove small objects from the foreground)
    	cv::erode(redmask, redmask, erodeElement);
	cv::dilate(redmask, redmask, dilateElement);
	// morphological closing (fill small holes in the foreground)
  	dilate(redmask, redmask, dilateElement);
  	erode(redmask, redmask, erodeElement);
	// blur
	cv::blur(redmask,redmask,cv::Size(SENSITIVITY_VALUE, SENSITIVITY_VALUE));
	// threshold again to obtain binary image from blur output
	cv::threshold(redmask,redmask, SENSITIVITY_VALUE, 255, cv::THRESH_BINARY);
	cv::imshow("maskRED",redmask);	*/

	/*  // another method of detecting the object
	//searchForMovement(mask, frame);*/


	// find contours in the mask and initialize the current (x, y) center of the ball
	std::vector<std::vector<cv::Point> > contours;  // container for the contours
	std::vector<cv::Vec4i> hierarchy;

	// void findContours(InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset=Point())
	cv::findContours(mask.clone(), contours, hierarchy, // find the image contours
                     CV_RETR_EXTERNAL,
                     CV_CHAIN_APPROX_SIMPLE);

    	cv::Point2f center(-1000,-1000);                // define center. Set to arbitrary init value
    	missedPlayer = true;                            // flag to track the Player presence

	// only proceed if at least one contour was found (the contour correspond to the color blob dettected)
	if (contours.size() > 0){

		int largest_area=0;               // container for the max area
		int largest_contour_index=0;      // container for the index of the max area found in countours

		//find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
		for(int i=0; i < contours.size();i++){
			// iterate through each contour
			double a = cv::contourArea(cv::Mat(contours[i]),false);  //  Find the area of contour
            		// if the area is bigger than the already found one, update it
           		if(a > largest_area) {
				largest_area = a;
				largest_contour_index = i;
	    		}
		}

		cv::Point2f tempcenter;
  		float radius;
		// cv::minEnclosingCircle(points, center, radius)
		cv::minEnclosingCircle((cv::Mat)contours[largest_contour_index], tempcenter, radius);

		// cv::moments(array, bnaryImage)
		cv::Moments M = cv::moments((cv::Mat)contours[largest_contour_index]);
		center = cv::Point2f(int(M.m10 / M.m00), int(M.m01 / M.m00));

		// Only proceed if the radius meets a minimum size. This is used to restrict the size of the object detected
		if (radius > 15) {
			// draw the circle and centroid on the frame, then update the list of tracked points
			circle(frame, cv::Point(int(tempcenter.x), int(tempcenter.y)), int(radius), cv::Scalar(0, 255, 255), 2);
			circle(frame, center, 5, cv::Scalar(0, 0, 255), -1); // red point for center
		    	mainCenter = center;            // save the center of the detected circle
		    	missedPlayer = false;           // update the flag for the player presence
		}
	}

	// update the points queue
	pts.push_front(center);

	/* Loop over the set of tracked points (i.e, the history of points)
        in order to print a line in the frame representing the history of
        tracked points. This line color is set to RED */
	for (int i=1; i < (pts.size()-1); i++) {
		// ignore points that are NONE. (NONE = -1000)
		cv::Point2f ptback = pts[i - 1];
		cv::Point2f pt = pts[i];
		if ((ptback.x == -1000) or (pt.x == -1000))  continue;

		// otherwise, compute the thickness of the line and draw the connecting lines
		int thickness = int(sqrt(BUFFER / float(i + 1)) * 2.5);
		line(frame, pts[i - 1], pts[i], cv::Scalar(0, 0, 255), thickness);
		//putText(frame,"Tracking object at ", pt , 1, 1, cv::Scalar(255,0,0), 1);
	}

	src = frame.clone();   // update the input frame
	regmask = mask.clone(); // return value of mask
}


// another method to detect and search
void searchForMovement(cv::Mat thresholdImage, cv::Mat &cameraFeed) {
	bool objectDetected = false;
	cv::Mat temp;
	thresholdImage.copyTo(temp);
	//vectors for output of findContours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);// retrieves all contours
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
}



/* distanceFunction -- used to compute the similarity betweent the pixels.*/
bool distanceFunction(float a, float b, int threshold){
    if(abs(a-b) <= threshold) return true;
	else return false;
}

/* segmentDepth -- a function that implements a "Region Growing algorithm", which
 is defined here in a "Breadth-first search" manner.
	sX --> Seed Pixel x value (columns == width)
	sY --> Seed Pixel y value (rows == height)
	threshold --> the value to be used in the call to "distanceFunction" method. If distance
    is less than threshold then recursion proceeds, else stops.
*/
void segmentDepth(cv::Mat& input, cv::Mat& dst, cv::Mat& roiSeg, int sX, int sY, float& ci, int threshold) {
	long int pixels = 0;           		// segmented pixels counter variable.
	std::vector< std::vector<int> > reach;	// This is the binary mask for the segmentation.
	for (int i = 0; i < input.rows; i++){	//They are set to 0 at first. Since no pixel is assigned to the segmentation yet.
		reach.push_back(std::vector<int>(input.cols));
	}

	// Define the queue. NOTE: it is a BFS based algorithm.
	std::queue< std::pair<int,int> > seg_queue;

	// verify the depth value of the seed position.
	float &in_pxl_pos = input.at<float>(sY,sX);

    if(in_pxl_pos == 0) {
	cout << "THE SEED DEPTH VALUE IS ZERO!!!!!" << endl;
        //ROS_WARN_STREAM("THE SEED DEPTH VALUE IS ZERO!!!!!");
    } else if (missedPlayer) {
	cout << "PLAYER IS MISSING" << endl;
        ci = -1 ;                                       //Set the value indicating player missing.
    } else {
        dst.at<float>(sY,sX) = 255;                     // add seed to output image.

	cout << "sono qui!!!" << endl;

        // Mark the seed as 1, for the segmentation mask.
    	reach[sY][sX] = 1;
        pixels++;                                        // increase the pixel counter.

    	// init the queue witht he seed.
        seg_queue.push(std::make_pair(sY,sX));

        /* Loop over the frame, based on the seed adding a
            new pixel to the dst frame accordingly*/
    	while(!seg_queue.empty())
    	{
            /* pop values */
    		std::pair<int,int> s = seg_queue.front();
    		int x = s.second;
    		int y = s.first;
            seg_queue.pop();
            /* ... */

            /* The following "if" blocks analise the pixels incorporating them to the
                dst frame if they meet the threshold condition. */

    		// Right pixel
    		if((x + 1 < input.cols) && (!reach[y][x + 1]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y, x + 1),
                threshold)){
                reach[y][x+1] = true;
                seg_queue.push(std::make_pair(y, x+1));
    			float &pixel = dst.at<float>(y,x+1);
                pixel = 255;
    			pixels++;;

    		}

    		//Below Pixel
    		if((y+1 < input.rows) && (!reach[y+1][x]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y+1,x),
                threshold)){
    			reach[y + 1][x] = true;
    			seg_queue.push(std::make_pair(y+1,x));
    			dst.at<float>(y+1,x) = 255;
    			pixels++;;
    		}

    		//Left Pixel
    		if((x-1 >= 0) && (!reach[y][x-1]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y,x-1),
                threshold)){
    			reach[y][x-1] = true;
    			seg_queue.push(std::make_pair(y,x-1));
    			dst.at<float>(y,x-1) = 255;
    			pixels++;;
    		}

    		//Above Pixel
    		if((y-1 >= 0) && (!reach[y - 1][x]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y-1,x),
                threshold)){
    			reach[y-1][x] = true;
    			seg_queue.push(std::make_pair(y-1,x));
    			dst.at<float>(y-1,x) = 255;
    			pixels++;;
    		}

    		//Bottom Right Pixel
    		if((x+1 < input.cols) && (y+1 < input.rows) && (!reach[y+1][x+1]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y+1,x+1),
                threshold)){
    			reach[y+1][x+1] = true;
    			seg_queue.push(std::make_pair(y+1,x+1));
    			dst.at<float>(y+1,x+1) = 255;
    			pixels++;
    		}

    		//Upper Right Pixel
    		if((x+1 < input.cols) && (y-1 >= 0) && (!reach[y-1][x+1]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y-1,x+1),
                threshold)){
    			reach[y-1][x+1] = true;
    			seg_queue.push(std::make_pair(y-1,x+1));
    			dst.at<float>(y-1,x+1) = 255;
    			pixels++;
    		}

    		//Bottom Left Pixel
    		if((x-1 >= 0) && (y + 1 < input.rows) && (!reach[y+1][x-1]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y+1,x-1),
                threshold)){
    			reach[y+1][x-1] = true;
    			seg_queue.push(std::make_pair(y+1,x-1));
    			dst.at<float>(y+1,x-1) = 255;
    			pixels++;
    		}

    		//Upper left Pixel
    		if((x-1 >= 0) && (y-1 >= 0) && (!reach[y-1][x-1]) &&
                distanceFunction(input.at<float>(sY,sX), input.at<float>(y-1,x-1),
                threshold)){
    			reach[y-1][x-1] = true;
    			seg_queue.push(std::make_pair(y-1,x-1));
    			dst.at<float>(y-1,x-1) = 255;
    			pixels++;
    		}
    	}

        /* FROM THIS POINT ON: Initialization of supporting code for detecting the blob in the
            segmented frame. This is needed for drawing the minimum bounding rectangle
            for the detected blob, thus, enabling the "contraction index" feature
            calculation.*/
        std::vector<std::vector<cv::Point> > contours;
    	std::vector<cv::Vec4i> hierarchy;
        cv::Mat bwImage(input.size(),CV_8UC1);
        dst.convertTo(bwImage,CV_8U,255.0/(255-0));
    	findContours(bwImage, contours, hierarchy,
                     CV_RETR_EXTERNAL,
                     CV_CHAIN_APPROX_SIMPLE);

        /* Only proceed if at least one contour was found. NOTE: This is need for avoiding
            "seg fault". */
    	if (contours.size() > 0){

            cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
            cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(8,8));
            cv::erode(dst, dst, erodeElement);
            cv::dilate(dst, dst, dilateElement);

            int largest_area=0;
    	    int largest_contour_index=0;

    	    // find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
    	    for(int i=0; i < contours.size();i++){
    		// iterate through each contour.
                double a = cv::contourArea(cv::Mat(contours[i]),false);  //  Find the area of contour
    			if( a > largest_area){
    				largest_area=a;
    				largest_contour_index=i;   //Store the index of largest contour
    			}
    		}

            /*Calculate the minimum bounding box for the detected blob.*/
            cv::Rect rect = cv::boundingRect(contours[largest_contour_index]);
            cv::Point pt1, pt2;
            pt1.x = rect.x;
            pt1.y = rect.y;
            pt2.x = rect.x + rect.width;
            pt2.y = rect.y + rect.height;

            // Draws the rect in the dst image
            roiSeg = cv::Mat(dst,rect);
            cv::rectangle(dst, pt1, pt2, cv::Scalar(255,255,255), 2,6,0);// the selection green rectangle

            /* compute the Contraction index feature. It is the difference
                between the bounding rectangle area and the segmented object
                in the segmentation frame, normalized by
                the area of the rectangle. */
            int roiSegarea = roiSeg.total();
            ci = float(roiSegarea-pixels)/float(roiSegarea);
            /* ... */

            /* put the CI to frame */
            cv::putText(dst,
            std::to_string(ci),
            cv::Point(rect.x,rect.y-5), // Coordinates
            cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
            0.9, // Scale. 2.0 = 2x bigger
            cv::Scalar(255,255,255), // Color
            1 // Thickness
            ); // Anti-alias
            /*... */
        }
    }
}
