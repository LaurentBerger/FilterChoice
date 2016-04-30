#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>
#include <ctype.h>

using namespace cv;
using namespace std;


UMat GradientDericheY(UMat op, double alphaDerive, double alphaMoyenne);
UMat GradientDericheX(UMat op, double alphaDerive, double alphaMoyenne);
UMat GradientPaillouY(UMat op, double alphaDerive, double alphaMoyenne);
UMat GradientPaillouX(UMat op, double alphaDerive, double alphaMoyenne);
void CannyBis(  OutputArray _dst,
                double low_thresh, double high_thresh,
                bool L2gradient ,InputOutputArray _dx,InputOutputArray _dy);


static void DisplayImage(UMat x,string s)
{
	vector<Mat> sx;
	split(x, sx);
	vector<double> minVal(3), maxVal(3);
	for (int i = 0; i < static_cast<int>(sx.size()); i++)
	{
		minMaxLoc(sx[i], &minVal[i], &maxVal[i]);
	}
	maxVal[0] = *max_element(maxVal.begin(), maxVal.end());
	minVal[0] = *min_element(minVal.begin(), minVal.end());
	Mat uc;
	x.convertTo(uc, CV_8U,255/(maxVal[0]-minVal[0]),-255*minVal[0]/(maxVal[0]-minVal[0]));
	imshow(s, uc);
}

int aa = 100, ww = 10;
int aad = 10, aam = 10;
int lowThresholdDer = 20;
int maxThresholdDer = 20;
int lowThresholdPai = 20;
int maxThresholdPai = 20;
int lowThresholdCan = 20;
int maxThresholdCan = 20;
int aperture = 0;
int const max_lowThreshold = 1000;
Mat sobel_x, sobel_y;
UMat img;
const char* window_deriche = "Edge Map Deriche";
const char* window_paillou = "Edge Map Paillou";
const char* window_canny = "Edge Map Canny";

static void CannyThreshold(int, void*)
{
	Mat dst;
	double a = aa / 100.0, w = ww / 100.0;
	if (a<w)
		a = w + 0.1;
// PAILLOU
	UMat rx = GradientPaillouX(img, a, w);
	UMat ry = GradientPaillouY(img, a, w);
	double minv, maxv;
	minMaxLoc(rx, &minv, &maxv);
	minMaxLoc(ry, &minv, &maxv);
	Mat mm;
	mm = abs(rx.getMat(ACCESS_READ));
	rx.getMat(ACCESS_READ).convertTo(sobel_x, CV_16S, 1);
	mm = abs(ry.getMat(ACCESS_READ)); ry.getMat(ACCESS_READ).convertTo(sobel_y, CV_16S, 1);
	minMaxLoc(sobel_x, &minv, &maxv);
	minMaxLoc(sobel_y, &minv, &maxv);
	CannyBis(dst, lowThresholdPai, maxThresholdPai,  true, sobel_x, sobel_y);
	UMat modPai;
	add(rx.mul(rx), ry.mul(ry), modPai);
	sqrt(modPai, modPai);
	DisplayImage(modPai,"Paillou module");
// DERICHE
	double ad = aad / 100.0, am = aam / 100.0;
	imshow(window_paillou, dst);
	rx = GradientDericheX(img, ad, am);
	ry = GradientDericheY(img, ad, am);
	minMaxLoc(rx, &minv, &maxv);
	minMaxLoc(ry, &minv, &maxv);
	mm = abs(rx.getMat(ACCESS_READ));
	rx.getMat(ACCESS_READ).convertTo(sobel_x, CV_16S, 1);
	mm = abs(ry.getMat(ACCESS_READ)); ry.getMat(ACCESS_READ).convertTo(sobel_y, CV_16S, 1);
	minMaxLoc(sobel_x, &minv, &maxv);
	minMaxLoc(sobel_y, &minv, &maxv);
	CannyBis(dst, lowThresholdDer, maxThresholdDer, true, sobel_x, sobel_y);
	UMat modDer;
	add(rx.mul(rx), ry.mul(ry), modDer);
	sqrt(modDer, modDer);
	imshow(window_deriche, dst);
	DisplayImage(modDer, "Deriche module");
// CANNY
	Canny(img, dst, lowThresholdCan, maxThresholdCan, 2*aperture+3, true);
	imshow(window_canny, dst);

}


int main(int argc, char* argv[])
{
	cv::ocl::setUseOpenCL(false);
	//imread("c:/lib/opencv/samples/data/pic3.png", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
	//imread("f:/lib/opencv/samples/data/aero1.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
	//imread("C:/Users/Laurent.PC-LAURENT-VISI/Downloads/14607367432299179.png", CV_LOAD_IMAGE_COLOR).copyTo(m);
//	imread("C:/Users/Laurent.PC-LAURENT-VISI/Desktop/n67ut.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
//	imread("C:/Users/Laurent.PC-LAURENT-VISI/Desktop/n67ut.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(img);
//	imread("c:/lib/opencv/samples/data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(m);
//	imread("c:/lib/opencv/samples/data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(img);
//	imread("f:/lib/opencv/samples/data/pic2.png", CV_LOAD_IMAGE_GRAYSCALE).copyTo(img);
	imread("A9MKM.jpg", CV_LOAD_IMAGE_GRAYSCALE).copyTo(img);
    
	namedWindow(window_deriche, WINDOW_AUTOSIZE);
	namedWindow(window_paillou, WINDOW_AUTOSIZE);
	namedWindow(window_canny, WINDOW_AUTOSIZE);

      /// Create a Trackbar for user to enter threshold
	createTrackbar("Der. Min.", window_deriche, &lowThresholdDer, 500, CannyThreshold);
	createTrackbar("Der. Max.", window_deriche, &maxThresholdDer, 500, CannyThreshold);
	createTrackbar("Der. a:", window_deriche, &aad, 400, CannyThreshold);
	createTrackbar("Der. b:", window_deriche, &aam, 400, CannyThreshold);
	createTrackbar("Pai. Min:", window_paillou, &lowThresholdPai, max_lowThreshold, CannyThreshold);
	createTrackbar("Pai. Max.", window_paillou, &maxThresholdPai, max_lowThreshold, CannyThreshold);
	createTrackbar("Pai. a", window_paillou, &aa, 400, CannyThreshold);
	createTrackbar("Pai. w:", window_paillou, &ww, 400, CannyThreshold);
	createTrackbar("Can. Min.", window_canny, &lowThresholdCan, max_lowThreshold, CannyThreshold);
	createTrackbar("Can. Max.", window_canny, &maxThresholdCan, max_lowThreshold, CannyThreshold);
	createTrackbar("Can. ape.", window_canny, &aperture, 2, CannyThreshold);
	CannyThreshold(0, NULL);
	waitKey();

    return 0;
}  

