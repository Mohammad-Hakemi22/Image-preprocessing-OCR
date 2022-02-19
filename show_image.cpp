#include "show_image.h"

using namespace std;
using namespace cv;


Mat show_image(string path)
{

	Mat img = imread(path);
	cvtColor(img, img, COLOR_BGR2GRAY);
	//namedWindow("image", WINDOW_NORMAL);
	if (img.empty())
	{
		cout << "no image";
	}
	else
	{
		//imshow("image", img);
		//waitKey(0);
	}

	return img;
}


Mat show_image_mat(Mat img, string window_name)
{
	namedWindow(window_name, WINDOW_NORMAL);
	if (img.empty())
	{
		cout << "no mat image";
	}
	else
	{
		imshow(window_name, img);
		waitKey(0);
	}

	return img;
}