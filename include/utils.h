/* utils.h
 */

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <queue>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

/* */
static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, cv::Mat& aux, int step, double, const cv::Scalar& color);

/* trackUser -- Function used to track color blobs on a RGB image. */
void trackUser(cv::Mat& src, cv::Mat& regmask);

/* testing. */
void searchForMovement(cv::Mat src, cv::Mat &cameraFeed);

/* resize -- A helper function to resize image. Here, the width is
default to 512 as it is the width of a kinect frame. That is the only
reason for doing it.*/
extern void resize(cv::Mat& image, cv::Mat& dst, int width, int height);

/* writeMatToFile -- a function to save a cv::Mat to file. Mainly
used for frame saving purposes.*/
void writeMatToFile(cv::Mat& m, const char* filename);

/* printMatImage -- a function to print a cv::Mat to console. It
Basically serve as for debugging purposes*/
void printMatImage(cv::Mat _m);

/* distanceFunction -- used to compute the similarity betweent the pixels.*/
bool distanceFunction(float a, float b, int threshold);

/* segmentDepth -- a function that implements a "Region Growing algorithm", which
 is defined here in a "Breadth-first search" manner.
	sX --> Seed Pixel x value (columns == width)
	sY --> Seed Pixel y value (rows == height)
	threshold --> the value to be used in the call to "distanceFunction" method. If distance
    is less than threshold then recursion proceeds, else stops.
*/
void segmentDepth(cv::Mat& input, cv::Mat& dst, cv::Mat& roiSeg, int sX, int sY, float& ci, int threshold);

#endif
