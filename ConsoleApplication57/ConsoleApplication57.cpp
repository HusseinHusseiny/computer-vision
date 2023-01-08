#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <Windows.h>
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/videoio/legacy/constants_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/video/background_segm.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>


using namespace cv;
using namespace std;


bool R1(int R, int G, int B) {
	bool e1 = (R > 95) && (G > 40) && (B > 20) && ((max(R, max(G, B)) - min(R, min(G, B))) > 15) && (abs(R - G) > 15) && (R > G) && (R > B);
	//bool e2 = (R > 220) && (G > 210) && (B > 170) && (abs(R - G) <= 15) && (R > B) && (G > B);
	return (e1);
}
Mat GetSkin(Mat const& src) {
	// allocate the result matrix
	Mat dst = src.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			// apply rgb rule
			bool a = R1(R, G, B);
			if (a)
				dst.ptr<Vec3b>(i)[j] = cwhite;
			else
				dst.ptr<Vec3b>(i)[j] = cblack;
		}
	}
	return dst;
}

int getContourAndHull(cv::Mat);
vector<int> elimNeighborHulls(vector<int>, vector<Point>); // to remove neighbor hulls
vector<int> filterHulls(vector<int>, vector<Point>, RotatedRect); // to remove hulls below a height
vector<int> filterHulls2(vector<int>, vector<Point>, vector<Point>, RotatedRect); // to further removehulls around palm

vector<Point> filterDefects(vector<Point>, RotatedRect); // to remove defects below a height
void findConvexityDefects(vector<Point>&, vector<int>&, vector<Point>&);

int main() {
	int ans, model;
	cout << "Choose which model you want " << endl << endl;
	cout << " '1' For HSV " << endl;
	cout << " '2' For YCRCB " << endl;
	cout << " '3' For RGB " << endl << endl;

	cin >> model;
	cout << "would you like to apply correction to the model" << endl;
	cout << "if yes press '1'" << endl;
	cout << "if no press '0'" << endl;
	cin >> ans;
	// Init background substractor
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2(500, 16.0, true);
	// Create empy input img, foreground and background image and foreground mask.
	Mat  foregroundMask, backgroundImage, foregroundImg;


	Mat image;

	Mat  skin, hsv, mask, ycrcb, mask2, rgb, mask3;
	char a[40];
	int count = 0;

	VideoCapture cap(0);

	if (!cap.isOpened()) {

		cout << "cannot open camera";

	}

	while (true) {

		cap >> image;

		cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
		double alpha = 2.2; /*< Simple contrast control */
		int beta = 0;       /*< Simple brightness control */

		for (int y = 0; y < image.rows; y++) {
			for (int x = 0; x < image.cols; x++) {
				for (int c = 0; c < image.channels(); c++) {
					new_image.at<cv::Vec3b>(y, x)[c] =
						cv::saturate_cast<uchar>(alpha * image.at<cv::Vec3b>(y, x)[c] + beta);
				}
			}
		}
		if (foregroundMask.empty()) {
			foregroundMask.create(image.size(), image.type());
		}
		// compute foreground mask 8 bit image
	  // -1 is parameter that chose automatically your learning rate
		bg_model->apply(image, foregroundMask, true ? -1 : 0);
		// smooth the mask to reduce noise in image
		GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
		// threshold mask to saturate at black and white values
		threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
		// create black foreground image
		foregroundImg = Scalar::all(0);
		// Copy source image to foreground image only in area with white mask
		image.copyTo(foregroundImg, foregroundMask);
		//Get background image
		bg_model->getBackgroundImage(backgroundImage);
		// Show the results
		//imshow("foreground mask", foregroundMask);




		if (model == 1 && ans == 0)
		{

			cvtColor(image, hsv, COLOR_BGR2HSV);


			Mat hsv_channels[3];
			/*cv::split(hsv, hsv_channels);
			imshow("HSV to gray", hsv_channels[2]);*/
			inRange(hsv, Scalar(0, 30, 80), Scalar(20, 150, 255), mask);
			GaussianBlur(mask, mask, Size(11, 11), 3.5, 3.5);
			int morph_size = 2;
			Mat element = getStructuringElement(
				MORPH_RECT, Size(2 * morph_size + 1,
					2 * morph_size + 1),
				Point(morph_size, morph_size));


			// For Erosion
			erode(mask, mask, element,
				Point(-1, -1), 1);

			// For Dilation
			dilate(mask, mask, element,
				Point(-1, -1), 1);


			//	cvAnd(foregroundMask, mask, finalmask, 0);
			Mat finalmask;
			cv::bitwise_and(foregroundMask, mask, finalmask);
			imshow("mask2", finalmask);
		}
		else if (model == 1 && ans == 1) {
			cvtColor(new_image, hsv, COLOR_BGR2HSV);


			Mat hsv_channels[3];
			/*cv::split(hsv, hsv_channels);
			imshow("HSV to gray", hsv_channels[2]);*/
			inRange(hsv, Scalar(0, 30, 80), Scalar(20, 150, 255), mask);
			GaussianBlur(mask, mask, Size(11, 11), 3.5, 3.5);
			int morph_size = 2;
			Mat element = getStructuringElement(
				MORPH_RECT, Size(2 * morph_size + 1,
					2 * morph_size + 1),
				Point(morph_size, morph_size));


			// For Erosion
			erode(mask, mask, element,
				Point(-1, -1), 1);

			// For Dilation
			dilate(mask, mask, element,
				Point(-1, -1), 1);

			//imshow("gammamask2", mask);
			Mat finalmask;
			cv::bitwise_and(foregroundMask, mask, finalmask);
			imshow("mask2", finalmask);

		}
		else if (model == 2 && ans == 0) {

			cvtColor(image, ycrcb, COLOR_BGR2YCrCb);

			inRange(ycrcb, Scalar(0, 133, 77), Scalar(255, 173, 127), mask2);

			GaussianBlur(mask2, mask2, Size(11, 11), 3.5, 3.5);
			int morph_size = 2;
			Mat element = getStructuringElement(
				MORPH_RECT, Size(2 * morph_size + 1,
					2 * morph_size + 1),
				Point(morph_size, morph_size));


			// For Erosion
			erode(mask2, mask2, element,
				Point(-1, -1), 1);

			// For Dilation
			dilate(mask2, mask2, element,
				Point(-1, -1), 1);
			//imshow("mask2", mask2);
			Mat finalmask;
			cv::bitwise_and(foregroundMask, mask2, finalmask);
			//vector<vector<Point>>contours; //Vector for storing contour
			//vector<Vec4i> hierarchy;
			//largest_contour_index = -1;
			//largest_area = 0;
			//Mat dst(image.rows, image.cols, CV_8UC1, Scalar::all(0));
			//findContours(finalmask.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

			//for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
			//{
			//	double a = contourArea(contours[i], false);  //  Find the area of contour
			//	if (a > largest_area) {
			//		largest_area = a;
			//		largest_contour_index = i;                //Store the index of largest contour
			//		bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
			//	}

			//}

			//if (largest_contour_index >= 0)
			//{
			//	Scalar color(255, 255, 255);
			//	drawContours(dst, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
			//	rectangle(image, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);
			//}
			//Scalar color(255, 255, 255);
			//	drawContours(dst, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
			//	rectangle(image, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);

			Mat result;
			vector<vector<Point> >contours;
			vector<Vec4i>hierarchy;
			int savedContour = -1;
			double maxArea = 0.0;
			findContours(finalmask.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());

			for (int i = 0; i < contours.size(); i++)
			{
				double area = contourArea(contours[i]);
				if (area > maxArea)
				{
					maxArea = area;
					savedContour = i;
				}
			}
			drawContours(result, contours, savedContour, Scalar(255), CV_FILLED, 8);
			finalmask &= result;

			imshow("mask2", finalmask);

		}
		else if (model == 2 && ans == 1) {
			cvtColor(new_image, ycrcb, COLOR_BGR2YCrCb);

			inRange(ycrcb, Scalar(0, 133, 77), Scalar(255, 173, 127), mask2);

			GaussianBlur(mask2, mask2, Size(11, 11), 3.5, 3.5);
			int morph_size = 2;
			Mat element = getStructuringElement(
				MORPH_RECT, Size(2 * morph_size + 1,
					2 * morph_size + 1),
				Point(morph_size, morph_size));


			// For Erosion
			erode(mask2, mask2, element,
				Point(-1, -1), 1);

			// For Dilation
			dilate(mask2, mask2, element,
				Point(-1, -1), 1);
			//imshow("gammamask2", mask2);

			Mat finalmask;
			cv::bitwise_and(foregroundMask, mask2, finalmask);
			Rect bounding_rect;
			Rect bounding_rect2;
			Rect bounding_rect3;
			Rect bounding_rect4;
			Rect bounding_rect5;
			Rect bounding_rect6;
			Mat mt(finalmask);
			Mat dst(mt.rows, mt.cols, CV_8UC1, Scalar::all(0));
			//Vector for storing contour 
			vector<vector<Point>> contoursb;
			vector<Vec4i> hierarchyb;
			vector<int> small_blobs;
			int contour_area;
			int threshold = 2000;
			//Find the contours in the image
			Mat binary_image;
			mt.copyTo(binary_image);
			findContours(mt, contoursb, hierarchyb, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			for (size_t i = 0; i < contoursb.size(); i++)
			{
				contour_area = contourArea(contoursb[i]);
				if (contour_area < threshold)
				{
					small_blobs.push_back(i);
				}
			}

			// fill-in all small contours with zeros
			for (size_t i = 0; i < small_blobs.size(); i++)
			{
				drawContours(binary_image, contoursb, small_blobs[i], Scalar(0, 0, 0), CV_FILLED, 8);
			}
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(binary_image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			Mat biggest(mt.rows, mt.cols, CV_8UC1, Scalar::all(0));

			int largestContour = 0;
			int secondLargestContour = 0;
			int thirdLargestContour = 0;
			int largestIndex = 0;
			int secondLargestIndex = 0;
			int thirdLargestIndex = 0;

			Scalar color(255, 255, 255);
			Scalar colorr(255, 0, 0);
			Scalar colorrr(0, 255, 0);
			int dy = 70;
			int dw = 6000;
			int n;
			int p;
			if (contours.size() >= 3)
			{
				for (int i = 0; i < contours.size(); i++)
				{
					if (contours[i].size() > largestContour)
					{

						thirdLargestContour = secondLargestContour;
						thirdLargestIndex = secondLargestIndex;

						secondLargestContour = largestContour;
						secondLargestIndex = largestIndex;

						largestContour = contours[i].size();
						largestIndex = i;

						bounding_rect3 = boundingRect(contours[thirdLargestIndex]);
						bounding_rect2 = boundingRect(contours[secondLargestIndex]);
						bounding_rect = boundingRect(contours[largestIndex]);
					}


					else if (contours[i].size() > secondLargestContour)
					{

						thirdLargestContour = secondLargestContour;
						thirdLargestIndex = secondLargestIndex;

						secondLargestContour = contours[i].size();
						secondLargestIndex = i;

						bounding_rect3 = boundingRect(contours[thirdLargestIndex]);
						bounding_rect2 = boundingRect(contours[secondLargestIndex]);
						bounding_rect = boundingRect(contours[largestIndex]);
					}

					else if (contours[i].size() > thirdLargestContour)
					{

						thirdLargestContour = contours[i].size();
						thirdLargestIndex = i;

						bounding_rect3 = boundingRect(contours[thirdLargestIndex]);
						bounding_rect2 = boundingRect(contours[secondLargestIndex]);
						bounding_rect = boundingRect(contours[largestIndex]);
					}
				}


				Point center = Point(bounding_rect.x, bounding_rect.y);
				Point center2 = Point(bounding_rect2.x, bounding_rect2.y);
				Point center3 = Point(bounding_rect3.x, bounding_rect3.y);
				int area1 = bounding_rect.width * bounding_rect.height;
				int area2 = bounding_rect2.width * bounding_rect2.height;
				int area3 = bounding_rect3.width * bounding_rect3.height;


				drawContours(biggest, contours, largestIndex, color, CV_FILLED, 8, hierarchy);
				drawContours(biggest, contours, secondLargestIndex, color, CV_FILLED, 8, hierarchy);
				drawContours(biggest, contours, thirdLargestIndex, color, CV_FILLED, 8, hierarchy);
				imshow("Biggest contours", biggest);



				if ((center3.y < center.y) && (center3.y < center2.y))
				{
					if ((abs(center.y - center2.y) <= dy) && (abs(area1 - area2) <= dw))
					{
						bounding_rect4 = bounding_rect;
						bounding_rect5 = bounding_rect2;
						drawContours(dst, contours, largestIndex, color, CV_FILLED, 8, hierarchy);
						drawContours(dst, contours, secondLargestIndex, color, CV_FILLED, 8, hierarchy);
						p = 1;
					}

					else
					{
						if (area2 < area1)
						{
							bounding_rect6 = bounding_rect;
							drawContours(dst, contours, largestIndex, color, CV_FILLED, 8, hierarchy);
						}

						if (area2 > area1)
						{
							bounding_rect6 = bounding_rect2;
							drawContours(dst, contours, secondLargestIndex, color, CV_FILLED, 8, hierarchy);
						}

						p = 0;
					}
				}
				if ((center2.y < center.y) && (center2.y < center3.y))
				{

					if ((abs(center.y - center3.y) <= dy) && (abs(area1 - area3) <= dw))
					{
						bounding_rect4 = bounding_rect;
						bounding_rect5 = bounding_rect3;
						drawContours(dst, contours, largestIndex, color, CV_FILLED, 8, hierarchy);
						drawContours(dst, contours, thirdLargestIndex, color, CV_FILLED, 8, hierarchy);
						p = 1;
					}

					else
					{
						if (area3 < area1)
						{
							bounding_rect6 = bounding_rect;
							drawContours(dst, contours, largestIndex, color, CV_FILLED, 8, hierarchy);
						}

						if (area3 > area1)
						{
							bounding_rect6 = bounding_rect3;
							drawContours(dst, contours, thirdLargestIndex, color, CV_FILLED, 8, hierarchy);
						}

						p = 0;
					}
				}

				if ((center.y < center2.y) && (center.y < center3.y))
				{
					if ((abs(center2.y - center3.y) <= dy) && (abs(area3 - area2) <= dw))
					{
						bounding_rect4 = bounding_rect2;
						bounding_rect5 = bounding_rect3;
						drawContours(dst, contours, secondLargestIndex, color, CV_FILLED, 8, hierarchy);
						drawContours(dst, contours, thirdLargestIndex, color, CV_FILLED, 8, hierarchy);
						p = 1;
					}

					else
					{

						if (area2 > area3)
						{
							bounding_rect6 = bounding_rect2;
							drawContours(dst, contours, secondLargestIndex, color, CV_FILLED, 8, hierarchy);
						}

						if (area2 < area3)
						{
							bounding_rect6 = bounding_rect3;
							drawContours(dst, contours, thirdLargestIndex, color, CV_FILLED, 8, hierarchy);
						}

						p = 0;
					}
				}
			}

			if (contours.size() == 2)
			{

				for (int i = 0; i < contours.size(); i++)
				{
					if (contours[i].size() > largestContour)
					{
						secondLargestContour = largestContour;
						secondLargestIndex = largestIndex;

						largestContour = contours[i].size();
						largestIndex = i;

						bounding_rect2 = boundingRect(contours[secondLargestIndex]);
						bounding_rect = boundingRect(contours[largestIndex]);
					}


					else if (contours[i].size() > secondLargestContour)
					{
						secondLargestContour = contours[i].size();
						secondLargestIndex = i;

						bounding_rect2 = boundingRect(contours[secondLargestIndex]);
						bounding_rect = boundingRect(contours[largestIndex]);
					}
				}
				Point center = Point(bounding_rect.x, bounding_rect.y);
				Point center2 = Point(bounding_rect2.x, bounding_rect2.y);

				int area1 = bounding_rect.width * bounding_rect.height;
				int area2 = bounding_rect2.width * bounding_rect2.height;

				drawContours(biggest, contours, largestIndex, color, CV_FILLED, 8, hierarchy);
				drawContours(biggest, contours, secondLargestIndex, color, CV_FILLED, 8, hierarchy);
				imshow("Biggest contours", biggest);

				if ((center.y < center2.y))
				{
					bounding_rect6 = bounding_rect2;
					drawContours(dst, contours, secondLargestIndex, color, CV_FILLED, 8, hierarchy);
					p = 0;
				}

				if ((center2.y < center.y))
				{
					bounding_rect6 = bounding_rect;
					drawContours(dst, contours, largestIndex, color, CV_FILLED, 8, hierarchy);
					p = 0;
				}

			}
			//imshow("Largest Contours / dst",dst);
			vector<vector<Point>> contourss;
			vector<Vec4i> hierarchyy;

			Mat dstt(dst.rows, dst.cols, CV_8UC1, Scalar::all(0));

			//Find the contours in the image
			findContours(dst, contourss, hierarchyy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

			for (int i = 0; i < contourss.size(); i++)
			{
				//drawContours( dstt, contourss, i, colorr, 1, 8, vector<Vec4i>(), 0, Point());
				drawContours(dstt, contourss, i, color, CV_FILLED, 8, hierarchyy);
			}
			//imshow("awta one or two contours/ hand(s)",dstt);

			Mat dss;
			cvtColor(dstt, dss, CV_GRAY2BGR, 3);

			// convex points
			vector<int> hull;

			// filtered convex points
			vector<int> filteredHulls;

			// concave points
			vector<Point> defects;

			// filtered concave points
			vector<Point> filteredDefects;

			vector<vector<Point>> hulls(contourss.size());
			vector<vector<Vec4i>> convdefect(contourss.size());

			// to obtain polygon
			vector<vector<Point>> approxContour(contourss.size());

			int r = 5;
			int d = 0;
			int h = 0;

			// to roughly find the center of palm
			RotatedRect minRect;
			double approxPolyDist = 15;

			//for (int i = 0; i < contourss.size(); i++)
			//{

			//	approxPolyDP(contourss[i], approxContour[i], approxPolyDist, false);
			//	contourss[i] = approxContour[i];

			//	minRect = minAreaRect(contourss[i]);
			//	convexHull(contourss[i], hull, false, false);
			//	convexHull(contourss[i], hulls[i], false);
			//	convexityDefects(contourss[i], hull, defects);

			//	// filter convex
			//	//filteredDefects = defects; // assign in case no filtering
			//	filteredDefects = filterDefects(defects, minRect);
			//	filteredHulls = hull;
			//	filteredHulls = filterHulls(hull, contourss[i], minRect);
			//	filteredHulls = elimNeighborHulls(filteredHulls, contourss[i]);
			//	filteredHulls = filterHulls2(filteredHulls, filteredDefects, contourss[i], minRect);
			//	int j;

			//	//draw polygon
			//	//fillConvexPoly(dstt, contourss[i], color, 8,0);
			//	//draw enclosing rectangle and center
			//	ellipse(dss, minRect, colorr, 1, 8);
			//	float ang = minRect.angle;
			//	printf("angle %f \n", ang);

			//	for (j = 0; j < filteredDefects.size(); j++) //blue
			//	{
			//		circle(dss, filteredDefects[j], r, colorr, 2, 8, 0);
			//		d++;
			//	}

			//	for (j = 0; j < filteredHulls.size(); j++) //green
			//	{
			//		circle(dss, contourss[i][filteredHulls[j]], r, colorrr, 1, 8, 0);
			//		h++;
			//	}
			//}
			//imshow("awta one or two contours/ hand(s)", dstt);
			//printf("Filtered Defects are %d \n", d);
			//printf("Filtered Hull are %d \n", h);
			//int s;
			//if (contourss.size() == 1)
			//{
			//	if ((3 < h) && (h < 6) && (2 < d) && (d < 6))
			//	{
			//		putText(dss, "One Hand Gesture", Point(40, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8, false);
			//		putText(dss, "State: Open", Point(210, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, 8, false);
			//		s = 1;
			//	}
			//	else
			//	{
			//		putText(dss, "No Hand Detected", Point(40, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8, false);
			//		s = 0;
			//	}
			//}
			//if (contourss.size() == 2)
			//{
			//	if ((6 < h) && (h < 11) && (4 < d) && (d < 13))
			//	{
			//		putText(dss, "Two Hands Gesture", Point(40, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8, false);
			//		putText(dss, "State: Open", Point(210, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, 8, false);
			//		s = 1;
			//	}
			//	else
			//	{
			//		putText(dss, "No Hand Detected", Point(40, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8, false);
			//		s = 0;
			//	}
			//}
			//if (contourss.size() == 0)
			//{
			//	putText(dss, "No Hand Detected", Point(40, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8, false);
			//	s = 0;
			//}
			////Draw Hull results
			//for (int i = 0; i < contourss.size(); i++)
			//{
			//	drawContours(dss, hulls, i, Scalar(0, 0, 255), 1, 8, vector<Vec4i>(), 0, Point());
			//}

						//	imshow("mask2", finalmask);

		}
		else if (model == 3 && ans == 0) {
			Mat mask3 = GetSkin(image);
			GaussianBlur(mask3, mask3, Size(11, 11), 3.5, 3.5);
			int morph_size = 2;
			Mat element = getStructuringElement(
				MORPH_RECT, Size(2 * morph_size + 1,
					2 * morph_size + 1),
				Point(morph_size, morph_size));


			// For Erosion
			erode(mask3, mask3, element,
				Point(-1, -1), 1);

			// For Dilation
			dilate(mask3, mask3, element,
				Point(-1, -1), 1);

			//imshow("original", image);
			//imshow("skin", mask3);
			Mat finalmask;
			cv::bitwise_and(foregroundMask, mask3, finalmask);
			imshow("mask2", finalmask);
		}
		else if (model == 3 && ans == 1) {
			Mat mask3 = GetSkin(new_image);
			GaussianBlur(mask3, mask3, Size(11, 11), 3.5, 3.5);
			int morph_size = 2;
			Mat element = getStructuringElement(
				MORPH_RECT, Size(2 * morph_size + 1,
					2 * morph_size + 1),
				Point(morph_size, morph_size));


			// For Erosion
			erode(mask3, mask3, element,
				Point(-1, -1), 1);

			// For Dilation
			dilate(mask3, mask3, element,
				Point(-1, -1), 1);

			//imshow("original", new_image);
			//imshow("skin", mask3);
			Mat finalmask;
			cv::bitwise_and(foregroundMask, mask3, finalmask);
			imshow("mask2", finalmask);

		}

		waitKey(25);

	}

	return 0;

}
void findConvexityDefects(vector<Point>& contour, vector<int>& hull, vector<Point>& convexDefects) {
	if (hull.size() > 0 && contour.size() > 0) {
		CvSeq* contourPoints;
		CvSeq* defects;
		CvMemStorage* storage;
		CvMemStorage* strDefects;
		CvMemStorage* contourStr;
		CvConvexityDefect* defectArray = 0;
		strDefects = cvCreateMemStorage();
		defects = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), strDefects);
		// transform our vector<Point> into a CvSeq* object of CvPoint.
		contourStr = cvCreateMemStorage();
		contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvSeq), sizeof(CvPoint),
			contourStr);
		for (int i = 0; i < (int)contour.size(); i++) {
			CvPoint cp = { contour[i].x, contour[i].y };
			cvSeqPush(contourPoints, &cp);
		}
		// do the same thing with the hull index
		int count = (int)hull.size();
		int* hullK = (int*)malloc(count * sizeof(int));
		for (int i = 0; i < count; i++) { hullK[i] = hull.at(i); }
		CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);
		// calculate convexity defects
		storage = cvCreateMemStorage(0);
		defects = cvConvexityDefects(contourPoints, &hullMat, storage);
		defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect) * defects->total);
		cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
		// store defects points in the convexDefects parameter.
		for (int i = 0; i < defects->total; i++) {
			CvPoint ptf;
			ptf.x = defectArray[i].depth_point->x;
			ptf.y = defectArray[i].depth_point->y;
			convexDefects.push_back(ptf);
		}
		// release memory
		cvReleaseMemStorage(&contourStr);
		cvReleaseMemStorage(&strDefects);
		cvReleaseMemStorage(&storage);
	}
}
vector<int> elimNeighborHulls(vector<int> inputIndex, vector<Point> inputPoints) {
	vector<int> tempfilteredHulls;
	float distance;
	float distThreshold = 20;
	if (inputIndex.size() == 0) {
		return inputIndex; // it's empty
	}
	if (inputIndex.size() == 1) {
		return inputIndex; // only one hull
	}
	for (unsigned int i = 0; i < inputIndex.size() - 1; i++) { // eliminate points that are close
		distance = sqrt((float)pow((float)inputPoints[inputIndex[i]].x - inputPoints[inputIndex[i + 1]].x, 2)
			+ pow((float)inputPoints[inputIndex[i]].y - inputPoints[inputIndex[i + 1]].y, 2));
		if (distance > distThreshold) { // set distance threshold to be 10
			tempfilteredHulls.push_back(inputIndex[i]);
		}
	}
	// get take of the last one, compare it with the first one
	distance = sqrt((float)pow((float)inputPoints[inputIndex[0]].x - inputPoints[inputIndex[inputIndex.size() -
		1]].x, 2) + pow((float)inputPoints[inputIndex[0]].y - inputPoints[inputIndex[inputIndex.size() - 1]].y, 2));
	if (distance > distThreshold) { // set distance threshold to be 10
		tempfilteredHulls.push_back(inputIndex[inputIndex.size() - 1]);
	}
	else if (inputIndex.size() == 2) { // the case when there are only two pts and they are together
		tempfilteredHulls.push_back(inputIndex[0]);
	}
	return tempfilteredHulls;
}
vector<int> filterHulls(vector<int> inputIndex, vector<Point> inputPoints, RotatedRect rect) {
	vector<int> tempFilteredHulls;
	float distThres = 20;
	for (unsigned int i = 0; i < inputIndex.size(); i++) {
		if (inputPoints[inputIndex[i]].y < (rect.center.y + distThres)) { // 10 being threshold height difference

			tempFilteredHulls.push_back(inputIndex[i]);
		}
	}
	return tempFilteredHulls;
}
vector<int> filterHulls2(vector<int> inputIndex, vector<Point> inputDefects, vector<Point> inputPoints,
	RotatedRect rect) {
	if (inputIndex.size() > 2 && inputDefects.size() > 1) {
		return inputIndex;
	}
	// only do filtering if there are less than 3 convex points
	vector<int> tempFilteredHulls;
	float palmRadius;
	if (rect.size.height <= rect.size.width) {
		palmRadius = (rect.size.height) / 2; // the normal case
	}
	else {
		palmRadius = (rect.size.width) / 2;
	}
	// for now ignore angle or rotation
	for (unsigned int i = 0; i < inputIndex.size(); i++) {
		if (inputPoints[inputIndex[i]].y < (rect.center.y - palmRadius)) { // 10 being threshold height difference
			tempFilteredHulls.push_back(inputIndex[i]);
		}
	}
	return tempFilteredHulls;
}
vector<Point> filterDefects(vector<Point> inputDefects, RotatedRect rect) {
	vector<Point> tempFilteredDefects;
	for (unsigned int i = 0; i < inputDefects.size(); i++) {
		if (inputDefects[i].y < (rect.center.y + 10)) {
			tempFilteredDefects.push_back(inputDefects[i]);
		}
	}
	return tempFilteredDefects;
}