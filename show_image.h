#pragma once
#ifndef SHOW_IMAGE_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define SHOW_IMAGE_H 

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Mat show_image(string path);
Mat show_image_mat(Mat img, string window_name);


#endif