#pragma once
#ifndef IMAGE1_H   // To make sure you don't declare the function more than once by including the header multiple times.
#define IMAGE1_H 

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/mat.hpp"
#include <fstream> 
#include <iostream>

using namespace std;
using namespace cv;

struct sum_varians_return
{
	float sum, var;
};

void show();
void gaussianBlur(Mat srcimage);
void medianBlur(Mat src_image);
void histogram(string const& name, Mat1b const& image);
void histogramEqualization(Mat src_image);
void threshold(Mat src_image);
void binarization(Mat src_image);
void sharpening(Mat src_image);
sum_varians_return sum_varians(int size, Mat src_image);
Mat addBorderToMat(Mat src, int size);
void kernel(Mat src, int size);
int reflect(int M, int x);
void refletedIndexing(Mat src);
void dilation(Mat src, int dilation_type = 0);
void erosion(Mat src, int erosion_type);
void morphology(Mat src);
void houghline(Mat src);

#endif
