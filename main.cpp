
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;

// Struct for a pixel strip
struct Strip {
	int length;
	int nStart;
	int nStop;
	cv::Point2f stripVecX;
	cv::Point2f stripVecY;
	cv::Mat data;
};

int subpixSampleSafe(const cv::Mat& pSrc, const cv::Point2f& p) {
	// floorf -> like int casting, but -2.3 will be the smaller number -> -3
	// Point is float, we want to know which color does it have
	int fx = int(floorf(p.x));
	int fy = int(floorf(p.y));

	if (fx < 0 || fx >= pSrc.cols - 1 ||
		fy < 0 || fy >= pSrc.rows - 1)
		return 127;

	// Slides 15
	int px = int(256 * (p.x - floorf(p.x)));
	int py = int(256 * (p.y - floorf(p.y)));

	// Here we get the pixel of the starting point
	unsigned char* i = (unsigned char*)((pSrc.data + fy * pSrc.step) + fx);

	// Shift 2^8
	// Internsity
	int a = i[0] + ((px * (i[1] - i[0])) >> 8);
	i += pSrc.step;
	int b = i[0] + ((px * (i[1] - i[0])) >> 8);

	// We want to return Intensity for the subpixel
	return a + ((py * (b - a)) >> 8);
}

// dx and dy are the x and y length of the subsections. st is the r
Strip calculateStrip(double dx, double dy) {
	Strip result;

	// make strip length proportional to subdivision length
	double length = sqrt(dx * dx + dy * dy);
	result.length = (int)(0.8 * length);
	if (result.length < 5)
		result.length = 5;

	// make strip length odd so it has a center pixel
	if (result.length % 2 == 0)
		result.length++;

	// shift half
	result.nStop = result.length / 2;
	result.nStart = -result.nStop;

	// sample a strip of width 3px
	cv::Size stripSize;
	stripSize.width = 3;
	stripSize.height = result.length;

	// normalize X direction vector
	result.stripVecX.x = dx / length;
	result.stripVecX.y = dy / length;

	// normalized Y direction vector, perpendicular to X
	result.stripVecY.x = result.stripVecX.y;
	result.stripVecY.y = -result.stripVecX.x;

	result.data = cv::Mat(stripSize, CV_8UC1);

	return result;
}

int main() {
	cout << "Starting AR Program" << endl;

	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	typedef vector<cv::Point> contour_t;
	typedef vector<contour_t> contour_vector_t;

	cout << "Starting webcam capture..." << endl;
	cv::VideoCapture capture(9);
	cv::Mat frame;
	cv::Mat modFrame;
	
	if (capture.isOpened())
		cout << "Success!" << endl << endl;

	else {
		cout << "Fail..." << endl;
		cout << "Loading Video instead" << endl;
		string videoPath = "MarkerMovie.MP4";
		capture = cv::VideoCapture(videoPath);

		if (capture.isOpened())
			cout << "Successfully loaded MarkerMovie.MP4" << endl;

		else {
			cout << "Could load neither webcam nor movie, exiting...";
			capture.release();
			exit(1);
		}
	}

	string outputWindowName = "Output";
	cv::namedWindow(outputWindowName);
	int trackbarPos = 80;
	cv::createTrackbar("thresholdTrackbar", outputWindowName, &trackbarPos, 255);

	cout << "Test";
	
	while (capture.read(frame)) {
		if (cv::waitKey(25) == 27 || !cv::getWindowProperty(outputWindowName, cv::WND_PROP_VISIBLE))
			break;

		// convert to grayscale
		cv::cvtColor(frame, modFrame, cv::COLOR_BGR2GRAY);

		// apply variable thrshold filter
		if (trackbarPos == 0)
			cv::adaptiveThreshold(modFrame, modFrame, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 33, 5);

		else
			cv::threshold(modFrame, modFrame, trackbarPos, 255, cv::THRESH_BINARY);

		/*
		// dilate and erode
		int size = 5;
		cv::Mat kernel = cv::Mat::ones(size, size, 0);
		cv::dilate(modFrame, modFrame, kernel);
		cv::erode(modFrame, modFrame, kernel);
		*/

		// search for contours
		contour_vector_t contours = {};
		cv::findContours(modFrame, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

		// filter contours
		for (size_t i = 0; i < contours.size(); i++) {

			contour_t approx_contour;

			cv::approxPolyDP(contours[i], approx_contour, cv::arcLength(contours[i], true) * 0.02, true);

			if (approx_contour.size() == 4) {
				// if polyline is a rectangle, draw contour on image
				cv::polylines(frame, approx_contour, true, cv::Scalar(0, 0, 255), 4);
				
				// subdivide each line into seven parts of equal length
				for (size_t j = 0; j < 4; j++) {
					cv::Point A = approx_contour[j];
					cv::Point B = approx_contour[(j + 1) % 4];

					double dx = ((double)(B.x - A.x)) / 7.0;
					double dy = ((double)(B.y - A.y)) / 7.0;
					Strip strip = calculateStrip(dx, dy);

					// draw the first circle and the 6 intermediate ones
					for (double k = 0; k < 7; k++) {
						double x = (1.0 - (k / 7)) * A.x + (k / 7) * B.x;
						double y = (1.0 - (k / 7)) * A.y + (k / 7) * B.y;

						if (k == 0) // render the outer points differently
							cv::circle(frame, cv::Point2d(x, y), 3, cv::Scalar(255, 0, 0), -1);

						else {	// for the inner points, render and also compute the strips
							cv::circle(frame, cv::Point2d(x, y), 2, cv::Scalar(0, 255, 0), -1);
							
							// loop over 3 pixel width
							for (int m = -1; m <= 1; m++) {
								// loop over height
								for (int n = strip.nStart; n <= strip.nStop; n++) {
									cv::Point2f subPixel;
									subPixel.x = x + (double)m * strip.stripVecX.x + (double)n * strip.stripVecY.x;
									subPixel.y = y + (double)m * strip.stripVecX.y + (double)n * strip.stripVecY.y;

									// draw subpixel on image
									cv::circle(frame, subPixel, 1, cv::Scalar(255, 255, 0), -1);

									// get subpixel intensity and save it
									int w = m + 1;
									int h = n + (strip.length >> 1);
									strip.data.at<uchar>(h, w) = (uchar)subpixSampleSafe(frame, subPixel);
								}
							}

							// use sobel operator to determine gradient of strip
							// ( -1 , -2, -1 )
							// (  0 ,  0,  0 )
							// (  1 ,  2,  1 )

							/*

							// exclude first and last row because the kernel won't fit
							vector<double> sobelValues(strip.length - 2);

							//iterate kernel from second row to the second-to-last row
							for (int l = 1; l < strip.length - 1; l++) {
								// take the intensity value from the strip
								unsigned char* stripPtr = &(strip.data.at<uchar>(l - 1, 0));

								// calculate the gradient for the first row
								double r1 = -stripPtr[0] - 2.0 * stripPtr[1] - stripPtr[2];

								// r2 = 0

								// calculate the gradient for the third row
								stripPtr += 2 * strip.data.step;
								double r3 = stripPtr[0] + 2.0 * stripPtr[1] + stripPtr[2];

								// write result into sobel value vector
								sobelValues[l - 1] = r1 + r3;
							}

							*/

							// simple sobel over the y direction
							cv::Mat grad_y;
							cv::Sobel(strip.data, grad_y, CV_8UC1, 0, 1);

							double maxIntensity = -1;
							int maxIntensityIndex = 0;

							// find max value
							for (int l = 0; l < strip.length - 2; ++l) {
								if (grad_y.at<uchar>(l, 1) > maxIntensity) {
									maxIntensity = grad_y.at<uchar>(l, 1);
									maxIntensityIndex = l;
								}
							}

							// Goal: fit second-degree polynomial

							double y0, y1, y2;

							// point before and after max intensity
							unsigned int max1 = maxIntensityIndex - 1;
							unsigned int max2 = maxIntensityIndex + 1;

							// if the index is at the border, then we are out of the strip and take 0
							y0 = (maxIntensityIndex <= 0) ? 0 : grad_y.at<uchar>(max1, 1);
							y1 = grad_y.at<uchar>(maxIntensityIndex, 1);
							y2 = (maxIntensityIndex >= strip.length - 3) ? 0 : grad_y.at<uchar>(max2, 1);

							// Formula for calculating the x-coordinate of the vertex of a parabola, given 3 points with equal distances 
							// (xv means the x value of the vertex, d the distance between the points): 
							// xv = x1 + (d / 2) * (y2 - y0)/(2*y1 - y0 - y2)

							// d = 1 because of normalization and x1 will be added later
							double pos = (y2 - y0) / (4 * y1 - 2 * y0 - 2 * y2);

							// if the pos is nan, there is no solution
							if (isnan(pos))
								continue;

							// exact point with subpixel accuracy
							cv::Point2d edgeCenter;

							int maxIndexShift = maxIntensityIndex - (strip.length >> 1);
							edgeCenter.x = (double)x + ((double)maxIndexShift + pos) * strip.stripVecY.x;
							edgeCenter.y = (double)y + ((double)maxIndexShift + pos) * strip.stripVecY.y;

							// highlight subpixel with blue color
							cv::circle(frame, edgeCenter, 2, CV_RGB(255, 0, 255), -1);

						}
					}
				}
			}
		}

		// output frame
		cv::imshow("Output", frame);
	}

	cv::destroyAllWindows();
	capture.release();
}