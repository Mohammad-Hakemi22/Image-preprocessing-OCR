#include <iostream>
#include "show_image.h"
#include <string>
#include <math.h> 
#include <fstream>
#include <numeric>


using namespace std;
using namespace cv;

struct sum_varians_return
{
	float avg, var;
};

struct stdblock
{
	int zero, non_zero;
	float zero_ratio, non_zero_ratio, ratio;
};


Mat otsuThreshold(Mat src);
void gaussianBlur(Mat);
void medianBlur(Mat src_image);
void histogram(string const& name, Mat1b const& image);
void histogramEqualization(Mat src_image);
void threshold(Mat src_image);
void binarization(Mat src_image);
void sharpening(Mat src_image);
sum_varians_return sum_varians(int size, Mat src_image);
Mat addBorderToMat(Mat src, int size);
void kernel(Mat src, int size);
void rect_kernel(Mat src, int size, int image_number);
int reflect(int M, int x);
void refletedIndexing(Mat src);
void dilation(Mat src, int dilation_type);
void erosion(Mat src, int erosion_type);
void morphology(Mat src);
void houghline(Mat src);
float variance(vector<int> ker, int avg);
int histogramSpredLine(Mat src);
Mat maxGray(Mat img);
void gsa(Mat image);
stdblock blockStd(Mat src, int blockSize);
int threshBlockStd(Mat src, int blockSize);
void areaThreshold(Mat src);
void checkBorderBadScanned(Mat src);
vector<float> kmeans_thresholding(Mat src);
void local_otsu(Mat src, int size, int image_number);
void kmeans_binarization(Mat src, int image_number);
int harlik(Mat src, int image_number);
int findMaxIndex(vector<int> hist, int max);
void localHarlik(Mat src, int size, int image_number);
double compute_skew(Mat src);
void blockSkewness(Mat src, int blockSize);
vector<Rect> detectLetters(Mat img_gray);
void textToImage();
void distroyImage(Mat src);
void AddGaussianNoise(Mat mSrc, double Mean, double StdDev);
void addSaltAndPaper(Mat src, float percent);
void randomLine(Mat src, int number);
void resizeImage(Mat src);


void show()
{
	Mat matImage;
	Mat matImage1;
	string image_path;
	stdblock result1;
	image_path = "####.jpg";
	matImage = show_image(image_path);
	//matImage = show_image(image_path);
	int j = 0;

	//for (int i = 2; i < 7; i++)
	//{
	//	image_path = "C:/Users/Mohammad/Downloads/Compressed/Data/test/" + to_string(i) + ".jpg";
	//	matImage = show_image(image_path);
	//	//kmeans_binarization(matImage, i);
	//	rect_kernel(matImage, 3, i);
	//	//local_otsu(matImage, 21,i);
	//	//harlik(matImage, i);
	//	//localHarlik(matImage, 10, i);
	//}

	//gaussianBlur(matImage);
	//medianBlur(matImage);
	//histogramEqualization(matImage);
	//threshold(matImage);
	//binarization(matImage);
	//otsuThreshold(matImage);
	//sharpening(matImage);
	//addBorderToMat(matImage, 13);
	//kernel(matImage, 3);
	//rect_kernel(matImage, 3);
	//refletedIndexing(matImage);
	//dilation(matImage, 0);
	//erosion(matImage, 0);
	//morphology(matImage);
	//houghline(matImage);
	//custome_kernel(matImage, 3);
	//j = histogramSpredLine(matImage);
	//cout << j;
	//matImage1=maxGray(matImage);
	//binarization(matImage1);
	//gsa(matImage);
	//result1 = blockStd(matImage, 5);
	//cout << "zero block : " << result1.zero << "\n" << "non zero block : " << result1.non_zero << "\n";
	//cout << "zero ratio block : " << result1.zero_ratio << "\n" << "non zero ratio block : " << result1.non_zero_ratio << "\n";
	//cout << "ratio : " << result1.ratio << "\n";

	//cout << threshBlockStd(matImage, 5);
	//Mat img = imread("####.jpg";);
	//areaThreshold(img);
	//checkBorderBadScanned(matImage);

	/*kmeans_thresholding(matImage);*/
	//kmeans_binarization(matImage);
	//harlik(matImage);
	//localHarlik(matImage, 11);
	local_otsu(matImage,21,16);
	//compute_skew(matImage);
	//blockSkewness(matImage, 500);
	//detectLetters(matImage);
	//textToImage();
	//distroyImage(matImage);
	//AddGaussianNoise(matImage, 0.0, 10.0);
	//addSaltAndPaper(matImage,5);
	//randomLine(matImage, 10);
	//resizeImage(matImage);
}

Mat otsuThreshold(Mat src)
{
	Mat dst;
	threshold(src, dst, 0, 255, THRESH_BINARY | THRESH_OTSU);
	show_image_mat(dst, " ");
	imwrite("####.jpg";, dst);
	return dst;
}

void gaussianBlur(Mat src_image)
{
	string name_of_window = "Gaussian Blured Image";
	Mat image_blurred_with_7x7_kernel;
	GaussianBlur(src_image, image_blurred_with_7x7_kernel, Size(107, 107), 0);
	show_image_mat(image_blurred_with_7x7_kernel, name_of_window);

	imwrite("####.jpg";, image_blurred_with_7x7_kernel);
}

void medianBlur(Mat src_image)
{
	string name_of_window = "Median Blured Image";
	Mat blurred_image;

	medianBlur(src_image, blurred_image, 7);
	imwrite("####.jpg";, blurred_image);
	show_image_mat(blurred_image, name_of_window);
}

void histogram(string const& name, Mat1b const& image)
{

	int bins = 256;
	int histSize[] = { bins };

	float lranges[] = { 0, 256 };
	const float* ranges[] = { lranges };

	Mat hist;
	int channels[] = { 0 };


	int const hist_height = 256;
	Mat3b hist_image = Mat3b::zeros(hist_height, bins);

	calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

	double max_val = 0;
	minMaxLoc(hist, 0, &max_val);


	for (int b = 0; b < bins; b++) {
		float const binVal = hist.at<float>(b);
		int   const height = cvRound(binVal * hist_height / max_val);
		line
		(hist_image
			, Point(b, hist_height - height), Point(b, hist_height)
			, Scalar::all(255)
		);
	}
	imwrite("####.jpg";, hist_image);
	imshow(name, hist_image);
	waitKey(0);
}


void histogramEqualization(Mat src_image)
{
	Mat dst;
	equalizeHist(src_image, dst);
	show_image_mat(dst, "Histogram Equalization");

}



//cv::ThresholdTypes{
//  cv::THRESH_BINARY = 0,
//  cv::THRESH_BINARY_INV = 1,
//  cv::THRESH_TRUNC = 2,
//  cv::THRESH_TOZERO = 3,
//  cv::THRESH_TOZERO_INV = 4,
//  cv::THRESH_MASK = 7,
//  cv::THRESH_OTSU = 8,
//  cv::THRESH_TRIANGLE = 16
//}

//cv::AdaptiveThresholdTypes{
//  cv::ADAPTIVE_THRESH_MEAN_C = 0,
//  cv::ADAPTIVE_THRESH_GAUSSIAN_C = 1
//}


void threshold(Mat src_image)
{
	Mat dst = Mat::zeros(src_image.size(), src_image.type());
	int thresholdType = 8;
	double maxValue = 255;
	int adaptiveMethod = 1;
	int blockSize = 5;
	double C = 1;
	int THRESHOTSU = 8;

	adaptiveThreshold(src_image, dst, maxValue, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 501, 10);
	imwrite("####.jpg";, dst);
	show_image_mat(dst, "Threshold");
}

void binarization(Mat src_image)
{
	int t = 200;
	Mat dst = Mat::zeros(src_image.size(), src_image.type());
	int max_val = 255;
	int min_val = 0;

	for (int i = 0; i < src_image.rows; i++)
	{
		for (int j = 0; j < src_image.cols; j++)
		{
			if (src_image.at<uchar>(i, j) < 170)
			{
				dst.at<uchar>(i, j) = min_val;
			}
			else
			{
				dst.at<uchar>(i, j) = max_val;
			}
		}
	}

	show_image_mat(dst, "binary");
}


void sharpening(Mat src_image)
{
	Mat kernel_sharping = Mat::zeros(Size(3, 3), CV_8SC1); //CV_8SC1 For Negative Numbers
	Mat dst = Mat::zeros(src_image.size(), src_image.type());
	Mat image_blurred_with_5x5_kernel;
	GaussianBlur(src_image, image_blurred_with_5x5_kernel, Size(3, 3), 0);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (i == 1 && j == 1)
			{
				kernel_sharping.at<uchar>(1, 1) = 9;
			}
			else
			{
				kernel_sharping.at<uchar>(i, j) = -1;
			}
		}
	}
	filter2D(image_blurred_with_5x5_kernel, dst, 0, kernel_sharping);
	//imwrite("####.jpg";, dst);
	//show_image_mat(dst, "sharp");

	//for (int i = 600; i < 900; i++)
	//{
	//	for (int j = 300; j < 500; j++)
	//	{
	//		printf("%d\t", dst.at<uchar>(i, j));
	//	}
	//}

}

sum_varians_return sum_varians(int size, Mat src_image)
{
	sum_varians_return s;
	float sum = 0;
	float avg = 0;
	float varians = 0;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			sum += src_image.at<uchar>(i, j);
		}
	}
	avg = sum / (size * size);

	for (int k = 0; k < size; k++)
	{
		for (int l = 0; l < size; l++)
		{
			varians = powf((src_image.at<uchar>(k, l) - avg), 2) / (size * size);
		}
	}

	s.avg = avg;
	s.var = varians;


	return s;
}

float variance(vector<int> ker, int avg)
{
	float var = 0;
	for (int i = 0; i < ker.size(); i++)
	{
		var += powf((ker.at(i) - avg), 2);
	}
	return var / ker.size();
}

Mat addBorderToMat(Mat src, int size)
{
	Mat new_src;
	int border = (size - 1) / 2;
	int top = border, bottom = border, left = border, right = border;
	int bordertype = BORDER_REPLICATE;
	copyMakeBorder(src, new_src, top, bottom, left, right, bordertype);

	return new_src;

}


int reflect(int M, int x)
{
	if (x < 0)
	{
		return -x - 1;
	}
	if (x >= M)
	{
		return 2 * M - x - 1;
	}
	return x;
}

vector<vector<int>> setKernel(int size, int type)
{

	if (size % 2 != 0)
	{
		cout << "even !";
	}

	vector<vector<int>> Kernel(size, vector<int>(size));

	switch (type)
	{
	case 0:
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{

				Kernel[i][j] = 1;

			}
		}

		break;

	default:
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{

				Kernel[i][j] = 1;

			}
		}
		break;
	}

	return Kernel;
}

void refletedIndexing(Mat src)
{
	float sum, x1, y1;
	Mat dst = Mat::zeros(src.size(), src.type());
	Mat ker = Mat::zeros(src.size(), src.type());
	//vector<vector<int>> Kernel(3, vector<int>(3));

	float Kernel[3][3] = {
								{1 , 1 , 1 },
								{1 , 1 , 1 },
								{1 , 1 , 1 }
	};

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			sum = 0.0;
			for (int k = -1; k <= 1; k++) {
				for (int j = -1; j <= 1; j++) {
					x1 = reflect(src.cols, x - j);
					y1 = reflect(src.rows, y - k);
					sum = sum + Kernel[j + 1][k + 1] * src.at<uchar>(y1, x1);
				}
			}
			dst.at<uchar>(y, x) = (sum / 9);

		}
	}

	show_image_mat(dst, "blur");

}

void kernel(Mat src, int size)
{

	Mat src_n;
	vector<int> kernel;
	//int* kernel = new int[size];
	float var = 0;
	int size_k = 0;
	size_k = (size - 1) / 2;
	src_n = addBorderToMat(src, size_k);
	Mat dst = Mat::zeros(src_n.size(), src.type());


	if (size % 2 == 0)
	{
		cout << "kernel size must be odd number !";
	}
	else
	{
		for (int i = 0; i < src_n.rows; i++)
		{
			for (int j = 0; j < src_n.cols; j++)
			{
				int sum = 0, avg = 0, t = size * size;
				kernel.clear();
				for (int m = i - size_k; m <= i + size_k; m++)
				{
					for (int n = j - size_k; n <= j + size_k; n++)
					{
						if (m < 0 || n < 0 || m > src_n.rows - 1 || n > src_n.cols - 1)
						{
							t--;
							continue;
						}
						sum += src_n.at<uchar>(m, n);
						kernel.insert(kernel.end(), src_n.at<uchar>(m, n));
					}
				}
				avg = sum / t;
				var = variance(kernel, avg);

				if (var > 0)
				{
					if (src_n.at<uchar>(i, j) > (avg + (-0.02 * sqrt(avg + (pow(var, 2))))))
					{
						dst.at<uchar>(i, j) = 255;
					}
					else
					{
						dst.at<uchar>(i, j) = 0;
					}
				}
				else
				{
					dst.at<uchar>(i, j) = src_n.at<uchar>(i, j);
				}

			}
		}



	}
	//for (int i = 0; i < kernel.size(); i++)
	//{
	//	cout << kernel.at(i) << " ";
	//}
	//cout << var;
	show_image_mat(dst, "threshold");
}



void rect_kernel(Mat src, int size, int image_number)
{
	Mat ROI;
	Scalar mean;
	Scalar std;
	Scalar max_mean;
	Scalar max_std;
	Mat src_n;
	src_n = addBorderToMat(src, size);
	Mat dst = Mat::ones(src_n.size(), src.type());

	int t = 0;
	double min, max, max_avg;
	for (int i = 0; i < src_n.rows - size; i++)
	{
		for (int j = 0; j < src_n.cols - size; j++)
		{
			Rect R = Rect(Point(j, i), Size(size, size));
			ROI = src_n(R);
			meanStdDev(ROI, mean, std);
			minMaxIdx(ROI, &min, &max);
			t = ((mean[0] + max)/2) * ((1 - 0.05) * (1 - (std[0] / 256)));
			if (t >= 255)
			{
				t = 0;
			}
			if (src_n.at<uchar>(i, j) >= t)
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	imwrite("####"; + to_string(image_number) + "xx.jpg", dst);
	cout << "image " + to_string(image_number) << " : " << "Done !" << "\n";
}


void dilation(Mat src, int dilation_type)
{
	Mat dst;

	if (dilation_type == 0)
	{
		dilation_type = MORPH_RECT;
	}
	else if (dilation_type == 1)
	{
		dilation_type = MORPH_CROSS;
	}
	else if (dilation_type == 2)
	{
		dilation_type = MORPH_ELLIPSE;
	}
	else
	{
		dilation_type = MORPH_RECT; //Default
	}
	dilate(src, dst, dilation_type);
	imwrite("####.jpg";, dst);
	show_image_mat(dst, "dilatation");
}


void erosion(Mat src, int erosion_type)
{
	Mat dst;

	if (erosion_type == 0)
	{
		erosion_type = MORPH_RECT;
	}
	else if (erosion_type == 1)
	{
		erosion_type = MORPH_CROSS;
	}
	else if (erosion_type == 2)
	{
		erosion_type = MORPH_ELLIPSE;
	}
	else
	{
		erosion_type = MORPH_RECT; //Default
	}
	erode(src, dst, erosion_type);
	imwrite("####.jpg";, dst);
	show_image_mat(dst, "erosion");
}


void morphology(Mat src)
{
	Mat dst;
	Mat dst1;
	Mat dst2;
	Mat dst3;
	Mat dst4;
	float kdata[] = { 0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0 };
	Mat kernel(5, 5, CV_8UC1, kdata);
	//morphologyEx(src, dst, MORPH_GRADIENT, kernel);
	//morphologyEx(src, dst1, MORPH_TOPHAT, kernel);
	//morphologyEx(src, dst2, MORPH_CLOSE, kernel);
	//morphologyEx(src, dst3, MORPH_OPEN, kernel);
	morphologyEx(src, dst4, MORPH_ELLIPSE, kernel);

	//imwrite("####.jpg";, dst);
	//imwrite("####.jpg";, dst1);
	//imwrite("####.jpg";, dst2);
	//imwrite("####.jpg";, dst3);
	imwrite("####.jpg";, dst4);

	//show_image_mat(dst, "gradian");
	////show_image_mat(dst1, "topHat");
	//show_image_mat(dst2, "CLOSE");
	//show_image_mat(dst3, "OPEN");

}


void houghline(Mat src)
{
	vector<Vec2f> lines;
	Mat dst, cdst, cdstP;
	Mat ker = Mat::zeros(Size(3, 3), src.type());

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (i == 1 && j == 1)
			{
				ker.at<uchar>(1, 1) = 8;
			}
			else
			{
				ker.at<uchar>(i, j) = -1;
			}
		}
	}
	//filter2D(src, dst, 0, ker);
	Canny(src, dst, 50, 200, 3);
	//dst = ~dst;

	cvtColor(dst, cdst, COLOR_GRAY2BGR);

	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);


	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 10000 * (-b));
		pt1.y = cvRound(y0 + 10000 * (a));
		pt2.x = cvRound(x0 - 10000 * (-b));
		pt2.y = cvRound(y0 - 10000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}


	imwrite("####.jpg";, cdst);
	show_image_mat(cdst, "line");
	//cout << lines.rows<<" * "<<lines.cols;
	//cout << lines.at<uchar>(2,0);

}


void transformHoughLine(vector<Vec2f> lines)
{
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
	}


}


int histogramSpredLine(Mat src)
{
	int size = 0;
	int max = 0;
	size = int(src.cols);
	vector<int> columns(size);
	Mat elements = Mat::zeros(src.size(), CV_8UC1);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				columns[j] += 1;
			}
		}
	}
	/*max = *max_element(columns.begin(), columns.end());*/

	for (int i = 0; i < columns.size(); i++)
	{
		elements.at<uchar>(0, i) = columns[i] * 200;
		elements.at<uchar>(1, i) = columns[i] * 200;
		elements.at<uchar>(2, i) = columns[i] * 200;
		elements.at<uchar>(3, i) = columns[i] * 200;
		elements.at<uchar>(4, i) = columns[i] * 200;
		elements.at<uchar>(5, i) = columns[i] * 200;
		elements.at<uchar>(6, i) = columns[i] * 200;
		elements.at<uchar>(7, i) = columns[i] * 20;
		elements.at<uchar>(8, i) = columns[i] * 20;
	}

	imwrite("####.jpg";, elements);
	//for (int j = 0; j < elements.cols; j++)
	//{
	//	printf("%d\t", elements.at<uchar>(0, j));
	//}

	bool flag = true;
	int k = 1;
	int l = 0;
	for (int j = 0; j < elements.cols - 1; j++)
	{
		if (elements.at<uchar>(0, j) > 0)
		{
			flag = true;
			l = j;
			while (flag)
			{
				if (elements.at<uchar>(0, l + 1) == 0)
				{
					k++;
					if (k >= 20)
					{
						return j;
					}
				}
				else
				{
					k = 0;
					flag = false;
				}
			}
		}
	}
}


Mat maxGray(Mat img)
{
	Mat channels[3];
	Mat result(img.rows, img.cols, CV_8UC1);

	split(img, channels);
	int b, g, r;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			b = channels[0].at<uchar>(i, j);
			g = channels[1].at<uchar>(i, j);
			r = channels[2].at<uchar>(i, j);
			result.at<uchar>(i, j) = max(b, max(g, r));
		}
	}
	show_image_mat(result, "re");
	return result;
}


void gsa(Mat src)
{
	Mat dst;
	Canny(src, dst, 50, 200, 3);
	int t = 50;
	int x, y = 0;

	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.at<uchar>(i, j) == 255)
			{
				x = i;
				y = j;
				for (int k = x - 1; k < x + 1; k++)
				{
					for (int l = y - 1; l < y + 1; l++)
					{
						if (k < 0 || l < 0 || k > src.rows - 1 || l > src.cols - 1)
						{
							t--;
							continue;
						}
					}
				}
			}
		}
	}

}


stdblock blockStd(Mat src, int blockSize)
{
	Mat imageBlock;
	vector<Mat> imageBlocks;
	vector<Scalar> blocksStd;
	Scalar mean;
	Scalar std;
	int zero_val = 0;
	int non_zero_val = 0;
	float zero_ratio = 0;
	float non_zero_ratio = 0;
	float ration = 0;
	stdblock result;
	//Laplacian(src_gray, dst, CV_16S, 1, 1, 0, BORDER_DEFAULT);
	Canny(src, src, 50, 200, 3);

	for (int i = 0; i < src.rows; i += blockSize)
	{
		for (int j = 0; j < src.cols; j += blockSize)
		{
			if (i + blockSize > src.rows && j + blockSize > src.cols)
			{
				imageBlock = src.rowRange(i, src.rows).colRange(j, src.cols);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std);
			}
			else if (i + blockSize > src.rows)
			{
				imageBlock = src.rowRange(i, src.rows).colRange(j, j + blockSize);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std);
			}
			else if (j + blockSize > src.cols)
			{
				imageBlock = src.rowRange(i, i + blockSize).colRange(j, src.cols);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std);
			}
			else
			{
				imageBlock = src.rowRange(i, i + blockSize).colRange(j, j + blockSize);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std[0]);
			}

		}
	}

	for (int i = 0; i < blocksStd.size(); i++)
	{
		if (blocksStd[i][0] == 0)
		{
			zero_val++;
		}
		else
		{
			non_zero_val++;
		}
	}
	result.zero = zero_val;
	result.non_zero = non_zero_val;
	zero_ratio = float(zero_val) / (src.rows * src.cols);
	non_zero_ratio = float(non_zero_val) / (src.rows * src.cols);
	result.zero_ratio = zero_ratio;
	result.non_zero_ratio = non_zero_ratio;
	result.ratio = zero_ratio / non_zero_ratio;
	return result;

}

int threshBlockStd(Mat src, int blockSize)
{
	Mat imageBlock;
	vector<Mat> imageBlocks;
	vector<Scalar> blocksStd;
	Scalar mean;
	Scalar std;
	int noise = 0;
	int text = 0;
	float zero_ratio = 0;
	float non_zero_ratio = 0;
	float ration = 0;
	stdblock result;
	Canny(src, src, 50, 200, 3);

	for (int i = 0; i < src.rows; i += blockSize)
	{
		for (int j = 0; j < src.cols; j += blockSize)
		{
			if (i + blockSize > src.rows && j + blockSize > src.cols)
			{
				imageBlock = src.rowRange(i, src.rows).colRange(j, src.cols);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std);
			}
			else if (i + blockSize > src.rows)
			{
				imageBlock = src.rowRange(i, src.rows).colRange(j, j + blockSize);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std);
			}
			else if (j + blockSize > src.cols)
			{
				imageBlock = src.rowRange(i, i + blockSize).colRange(j, src.cols);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std);
			}
			else
			{
				imageBlock = src.rowRange(i, i + blockSize).colRange(j, j + blockSize);
				meanStdDev(imageBlock, mean, std);
				blocksStd.insert(blocksStd.end(), std[0]);
			}

		}
	}

	for (int i = 0; i < blocksStd.size(); i++)
	{
		if (blocksStd[i][0] < 90.00 || blocksStd[i][0] > 130.00)
		{
			noise++;
		}
		else if (blocksStd[i][0] > 90.00 && blocksStd[i][0] < 130.00)
		{
			text++;
		}

	}

	return text;
}

void areaThreshold(Mat image)
{
	Mat gray_mat(image.size(), CV_8U);
	cvtColor(image, gray_mat, COLOR_BGR2GRAY);

	Mat image_th;
	Mat bin_mat(gray_mat.size(), gray_mat.type());
	adaptiveThreshold(gray_mat, image_th, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 3, 5);
	//threshold(gray_mat, image_th, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);


	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", image_th);
	waitKey(0);


	Mat labels;
	Mat stats;
	Mat centroids;
	connectedComponentsWithStats(image_th, labels, stats, centroids);


	cout << "stats.size()=" << stats.size() << endl;


	for (int i = 0; i < stats.rows; i++)
	{
		int x = stats.at<int>(Point(0, i));
		int y = stats.at<int>(Point(1, i));
		int w = stats.at<int>(Point(2, i));
		int h = stats.at<int>(Point(3, i));

		//std::cout << "x=" << x << " y=" << y << " w=" << w << " h=" << h << std::endl;

		Scalar color(0, 0, 200);
		Rect rect(x, y, w, h);
		rectangle(image, rect, color);
	}

	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", image);
	waitKey(0);
	imwrite("####.jpg";, image);

}

void checkBorderBadScanned(Mat src)
{
	int padding_row, padding_col, padding;
	padding_row = 0.1 * src.rows;
	padding_col = 0.1 * src.cols;
	//padding = int(min(padding_row, padding_col));
	padding = 30;

	Mat LE, RI, TO, DO;
	Rect R1 = Rect(Point(0, 0), Size(padding, src.rows));
	Rect R2 = Rect(Point(0, 0), Size(src.cols, padding));
	Rect R3 = Rect(Point(0, (src.rows - padding)), Size(src.cols, padding));
	Rect R4 = Rect(Point(src.cols - padding, 0), Size(padding, src.rows));
	LE = src(R1);
	TO = src(R2);
	DO = src(R3);
	RI = src(R4);
	countNonZero(LE);
	countNonZero(TO);
	countNonZero(DO);
	countNonZero(RI);
}

vector<float> kmeans_thresholding(Mat img)
{
	Mat label, centers;
	Mat points(Size(1, (img.rows * img.cols)), img.type());
	Mat locations(Size(2, (img.rows * img.cols)), img.type());
	vector<int> background;
	vector<int> forground;
	vector<int> gray;
	int c = 0;
	int max_forground = 0, min_background = 0;
	float b_avg, f_avg, g_avg;
	vector<float> result;
	int min1, min2, min3;


	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			points.at<uchar>(c, 0) = int(img.at<uchar>(i, j));
			locations.at<uchar>(c, 0) = i;
			locations.at<uchar>(c, 1) = j;
			c++;
		}
	}
	points.convertTo(points, CV_32F);

	double compactness = kmeans(points, 3, label, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.001), 10, KMEANS_PP_CENTERS, centers);
	label.convertTo(label, CV_8UC1);
	points.convertTo(points, CV_8UC1);
	centers.convertTo(centers, CV_8UC1);
	for (int i = 0; i < label.rows; i++)
	{
		if (int(label.at<uchar>(i, 0)) == 0)
		{
			forground.insert(forground.end(), int(points.at<uchar>(i, 0)));
		}
		else if (int(label.at<uchar>(i, 0)) == 2)
		{
			background.insert(background.end(), int(points.at<uchar>(i, 0)));
		}
		else
		{
			gray.insert(gray.end(), int(points.at<uchar>(i, 0)));
		}
	}


	//b_avg = accumulate(background.begin(), background.end(), 0.0) / (background.size());
	//f_avg = accumulate(forground.begin(), forground.end(), 0.0) / (forground.size());
	//g_avg = accumulate(gray.begin(), gray.end(), 0.0) / (gray.size());

	min1 = *min_element(background.begin(), background.end());
	min2 = *min_element(forground.begin(), forground.end());
	min3 = *min_element(gray.begin(), gray.end());
	//result.insert(result.end(), b_avg);
	//result.insert(result.end(), f_avg);
	//result.insert(result.end(), g_avg);
	result.insert(result.end(), min1);
	result.insert(result.end(), min2);
	result.insert(result.end(), min3);

	sort(result.begin(), result.end(), greater<float>());

	return result;
}

void local_otsu(Mat src, int size, int image_number)
{
	Mat ROI;
	Scalar mean;
	Scalar std;
	Scalar max_mean;
	Scalar max_std;
	Mat src_n;
	src_n = addBorderToMat(src, size);
	Mat dst = Mat::ones(src_n.size(), src.type());
	Mat dst1;
	int t = 0;
	double min, max, max_avg;

	for (int i = 0; i < src_n.rows - size; i++)
	{
		for (int j = 0; j < src_n.cols - size; j++)
		{
			Rect R = Rect(Point(j, i), Size(size, size));
			ROI = src_n(R);
			double t = threshold(ROI, dst1, 0, 255, THRESH_BINARY | THRESH_OTSU);

			if (src_n.at<uchar>(i, j) >= t)
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}

		}
	}
	imwrite("####"; + to_string(image_number) + ".jpg", dst);
	cout << "image " + to_string(image_number) << " : " << "Done !" << "\n";
}

void kmeans_binarization(Mat src, int image_number)
{
	vector<float> thresholds = kmeans_thresholding(src);
	Mat dst = Mat::zeros(src.size(), src.type());
	Scalar mean;
	Scalar std;

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) < thresholds[1])
			{
				dst.at<uchar>(i, j) = 0;

			}
			else if (src.at<uchar>(i, j) >= thresholds[0])
			{
				dst.at<uchar>(i, j) = 255;

			}
			else
			{
				int sum = 0, t = 0;
				float avg = 0;
				if (i - 1 >= 0 && j - 1 >= 0)
				{
					sum += int(src.at<uchar>(i - 1, j - 1));
					t++;
				}
				else if (i - 1 >= 0 && j >= 0)
				{
					sum += int(src.at<uchar>(i - 1, j));
					t++;
				}
				else if (i - 1 >= 0 && j + 1 <= src.cols)
				{
					sum += int(src.at<uchar>(i - 1, j + 1));
					t++;
				}
				else if (i >= 0 && j - 1 >= 0)
				{
					sum += int(src.at<uchar>(i, j - 1));
					t++;
				}
				else if (i >= 0 && j + 1 <= src.cols)
				{
					sum += int(src.at<uchar>(i, j + 1));
					t++;
				}
				else if (i + 1 <= src.rows && j - 1 >= 0)
				{
					sum += int(src.at<uchar>(i + 1, j - 1));
					t++;
				}
				else if (i + 1 <= src.rows && j >= 0)
				{
					sum += int(src.at<uchar>(i - 1, j));
					t++;
				}
				else if (i + 1 <= src.rows && j + 1 <= src.cols)
				{
					sum += int(src.at<uchar>(i + 1, j + 1));
					t++;
				}
				avg = sum / t;
				if (avg < thresholds[1])
				{
					dst.at<uchar>(i, j) = 0;
				}
				else if (avg >= thresholds[0])
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
	}

	imwrite("####"; + to_string(image_number) + ".jpg", dst);
	cout << "image " + to_string(image_number) << " : " << thresholds[0] << "," << thresholds[1] << "," << thresholds[2] << "\t" << "Done !" << "\n";
}


int findMaxIndex(vector<int> hist, int max)
{
	vector<int>::iterator it;
	int result;
	it = find(hist.begin(), hist.end(), max);
	if (it != hist.end())
	{
		result = it - hist.begin();
	}
	return result;
}

int harlik(Mat src, int image_number)
{
	Mat dst = Mat::zeros(src.size(), src.type());
	Mat grad;
	int ksize = 3;
	int ddepth = CV_16S;
	int hist[256] = {};
	int index = 0;
	int sum_s = 0;
	int sum_m = 0;
	int M = 0;
	vector<int> hist_u, shist_u;
	vector<int> hist_l, shist_l;
	int max1_l, max2_l, max1_u, max2_u;
	int imax1_l, imax2_l, imax1_u, imax2_u;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	vector<int>::iterator it_l, it_u;
	int T = 0;


	Sobel(src, grad_x, ddepth, 1, 0, ksize, BORDER_DEFAULT);
	Sobel(src, grad_y, ddepth, 0, 1, ksize, BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, grad);
	//show_image_mat(grad, " ");

	for (int i = 0; i < grad.rows; i++)
	{
		for (int j = 0; j < grad.cols; j++)
		{
			index = int(grad.at<uchar>(i, j));
			hist[index]++;

		}
	}

	for (int i = 0; i < sizeof(hist) / sizeof(int); i++)
	{
		if (i == 0)
		{
			sum_s += (1 * hist[i]);
			sum_m += hist[i];
		}
		else
		{
			sum_s += (i * hist[i]);
			sum_m += hist[i];
		}

	}

	M = (sum_s / sum_m);

	for (int i = 0; i < M; i++)
	{
		hist_l.insert(hist_l.end(), hist[i]);
	}

	for (int i = M; i < sizeof(hist) / sizeof(int); i++)
	{
		hist_u.insert(hist_u.end(), hist[i]);
	}

	shist_l = hist_l;
	shist_u = hist_u;

	sort(shist_l.begin(), shist_l.end(), greater<int>());
	sort(shist_u.begin(), shist_u.end(), greater<int>());

	max1_l = shist_l[0];
	max2_l = shist_l[1];
	max1_u = shist_u[0];
	max2_u = shist_u[1];


	imax1_l = (findMaxIndex(hist_l, max1_l) == 0 ? 1 : findMaxIndex(hist_l, max1_l));
	imax2_l = (findMaxIndex(hist_l, max2_l) == 0 ? 1 : findMaxIndex(hist_l, max2_l));
	imax1_u = findMaxIndex(hist_u, max1_u);
	imax2_u = findMaxIndex(hist_u, max2_u);

	T = ((imax1_l * max1_l) + (imax2_l * max2_l) + (imax1_u * max1_u) + (imax2_u * max2_u)) / (max1_l + max2_l + max1_u + max2_u);


	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) < T)
			{
				dst.at<uchar>(i, j) = 0;
			}
			else
			{
				dst.at<uchar>(i, j) = 255;
			}
		}
	}

	imwrite("####"; + to_string(image_number) + ".jpg", dst);
	cout << "Done !" << "\n";
	return T;
}


void localHarlik(Mat src, int size, int image_number)
{
	Mat ROI;
	Mat src_n;
	src_n = addBorderToMat(src, size);
	Mat dst = Mat::zeros(src_n.size(), src.type());
	int t = 0;

	for (int i = 10; i < src_n.rows - size; i++)
	{
		for (int j = 10; j < src_n.cols - size; j++)
		{
			Rect R = Rect(Point(j, i), Size(size, size));
			ROI = src_n(R);
			int t = harlik(ROI, 1);

			if (src_n.at<uchar>(i, j) >= t)
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	imwrite("####" + to_string(image_number) + ".jpg", dst);
	cout << "image " + to_string(image_number) << " : " << "Done !" << "\n";
}


double compute_skew(Mat src)
{
	Size size = src.size();
	src = ~src;
	vector<Vec4i> lines;
	HoughLinesP(src, lines, 1, CV_PI / 180, 100, size.width / 2.f, 20);
	Mat disp_lines(size, CV_8UC1, Scalar(0, 0, 0));
	double angle = 0.;
	unsigned nb_lines = lines.size();
	for (unsigned i = 0; i < nb_lines; ++i)
	{
		line(disp_lines, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255, 0, 0));
		angle += atan2((double)lines[i][3] - lines[i][1],
			(double)lines[i][2] - lines[i][0]);
	}
	angle /= nb_lines; // mean angle, in radians.

	//cout << angle * 180 / CV_PI << endl;



	//vector<Point> points;
	//Mat_<uchar>::iterator it = src.begin<uchar>();
	//Mat_<uchar>::iterator end = src.end<uchar>();
	//for (; it != end; ++it)
	//	if (*it)
	//		points.push_back(it.pos());
	//RotatedRect box = minAreaRect(Mat(points));
	//Mat rot_mat = getRotationMatrix2D(box.center, angle * 180 / CV_PI, 1);
	//Mat rotated;
	//warpAffine(src, rotated, rot_mat, src.size(), INTER_CUBIC);


	Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
	Mat rot = getRotationMatrix2D(center, angle * 180 / CV_PI, 1.0);

	Rect2f bbox = RotatedRect(Point2f(), src.size(), angle * 180 / CV_PI).boundingRect2f();

	rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;

	Mat dst;
	warpAffine(src, dst, rot, bbox.size());

	imwrite("####.jpg";, ~dst);
	return (angle * 180) / CV_PI;
}


void blockSkewness(Mat src, int blockSize)
{
	Mat imageBlock;

	for (int i = 0; i < src.rows; i += blockSize)
	{
		for (int j = 0; j < src.cols; j += blockSize)
		{
			if (i + blockSize > src.rows && j + blockSize > src.cols)
			{
				imageBlock = src.rowRange(i, src.rows).colRange(j, src.cols);
				compute_skew(imageBlock);

			}
			else if (i + blockSize > src.rows)
			{
				imageBlock = src.rowRange(i, src.rows).colRange(j, j + blockSize);
				compute_skew(imageBlock);
			}
			else if (j + blockSize > src.cols)
			{
				imageBlock = src.rowRange(i, i + blockSize).colRange(j, src.cols);
				compute_skew(imageBlock);
			}
			else
			{
				imageBlock = src.rowRange(i, i + blockSize).colRange(j, j + blockSize);
				compute_skew(imageBlock);
			}

		}
	}
}


vector<Rect> detectLetters(Mat img_gray)
{
	vector<Rect> boundRect;
	vector<int> width;
	vector<int> height;
	vector<int> x;
	vector<int> y;
	vector<double> angels;
	Mat img_sobel, img_threshold, element, ROI;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	threshold(img_sobel, img_threshold, 0, 255, THRESH_BINARY | THRESH_OTSU);
	element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);
	vector< vector< Point> > contours;
	findContours(img_threshold, contours, 0, 1);
	vector<vector<Point> > contours_poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
		if (contours[i].size() > 100)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			Rect appRect(boundingRect(Mat(contours_poly[i])));
			if (appRect.width > appRect.height && appRect.width >= 500 && appRect.width <= 1000 && appRect.height < 100)
			{
				boundRect.push_back(appRect);
				width.push_back(appRect.width);
				height.push_back(appRect.height);
				x.push_back(appRect.x);
				y.push_back(appRect.y);
			}

		}
	for (int i = 0; i < width.size(); i++)
	{
		Rect R = Rect(x[i] - 15, y[i] - 15, width[i] + 15, height[i] + 15);
		ROI = img_gray(R);
		angels.push_back(compute_skew(ROI));
	}

	for (size_t i = 0; i < angels.size(); i++)
	{
		cout << angels[i] << "\t";
	}
	//cout << "\n\n*******************************************\n\n";
	//for (size_t i = 0; i < boundRect.size(); i++)
	//{
	//	cout << boundRect[i] << "\t";
	//}

	//for (int i = 0; i < boundRect.size(); i++)
	//{
	//	rectangle(img_gray, boundRect[i], Scalar(0, 255, 0), 3, 8, 0);
	//}
	//imwrite("imgOut1.jpg", img_gray);
	/*show_image_mat(img_gray, " ");*/
	//imwrite("####.jpg"; , img_gray);
	return boundRect;
}

void textToImage()
{
	String str = "hello";
	int len = str.length();
	Mat image = Mat::zeros(Size(len + 200, 50), CV_8UC3);
	image = ~image;
	//for (int i = 0; i < 255; i++)
	//{

	//	image = image - Scalar::all(i);
	//}
	putText(image, "pic", Point(10, 20), FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 0, 0), 1, 8);
	show_image_mat(image, " ");

}

void distroyImage(Mat src)
{
	int total = 0;
	int x, y;
	total = src.cols * src.rows;

	for (int i = 0; i < int(total / 50); i++)
	{
		x = (rand() % src.rows);
		y = (rand() % src.cols);
		if (src.at<uchar>(x, y) > 120)
		{
			if (i % 3 == 0)
			{
				src.at<uchar>(x, y) = rand() % 255;
			}
		}
		else
		{
			src.at<uchar>(x, y) = (rand() % 150) + 15;
		}
	}
	//GaussianBlur(src, src, Size(3, 3),0);
	//medianBlur(src, src, 3);
	float kdata[] = { 0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0 };
	Mat kernel(5, 5, CV_8UC1, kdata);
	//morphologyEx(src, dst, MORPH_GRADIENT, kernel);
	morphologyEx(src, src, MORPH_TOPHAT, kernel);
	//show_image_mat(src, " ");
	imwrite("####.jpg";, ~src);
}


void AddGaussianNoise(Mat mSrc, double Mean = 0.0, double StdDev = 50.0)
{
	Mat mSrc_16SC = Mat(mSrc.size(), CV_8UC1);
	Mat mDst(mSrc.size(), mSrc.type());
	Mat mGaussian_noise = Mat(mSrc.size(), CV_8UC1);
	randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));

	mSrc.convertTo(mSrc_16SC, CV_8UC1);
	addWeighted(mSrc_16SC, 0.9, mGaussian_noise, 0.5, 0.0, mSrc_16SC);
	mSrc_16SC.convertTo(mDst, mSrc.type());

	show_image_mat(mDst, " ");

}

void addSaltAndPaper(Mat src, float percent)
{
	float total = 0, count = 0;
	int x, y;
	total = src.rows * src.cols;
	count = (percent / 100) * total;

	for (int i = 0; i < int(count); i++)
	{
		x = (rand() % src.rows);
		y = (rand() % src.cols);
		if (i % 2 == 0)
		{
			src.at<uchar>(x, y) = 255;
		}
		else
		{
			src.at<uchar>(x, y) = 0;
		}
	}

	show_image_mat(src, " ");

}


void randomLine(Mat src, int number)
{
	int x1, y1, x2, y2;
	Point pt1, pt2;
	RNG rng;

	for (int i = 0; i < number; i++)
	{
		pt1.y = rng.uniform(0, src.rows);
		pt1.x = rng.uniform(0, src.cols);
		pt2.y = rng.uniform(0, src.rows);
		pt2.x = rng.uniform(0, src.cols);
		if (i % 2 == 0)
		{
			line(src, pt1, pt2, Scalar(0, 0, 0), 1, 8);
		}
		else
		{
			line(src, pt1, pt2, Scalar(255, 255, 255), 1, 8);
		}
	}
	show_image_mat(src, " ");

}

void resizeImage(Mat src)
{
	Mat dst(src.size(), src.type());
	resize(src, dst, Size(round(0.5 * src.cols), round(0.5 * src.rows)), 0, 0);
	resize(dst, dst, Size(round(1 * src.cols), round(1 * src.rows)), 0, 0, INTER_CUBIC);
	show_image_mat(dst, " ");
}