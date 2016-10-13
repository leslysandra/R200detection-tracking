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

/* trackUser -- Function used to track color blobs on a RGB image. */
void trackUserByColor(cv::Mat& src, cv::Mat& regmask);

#endif
