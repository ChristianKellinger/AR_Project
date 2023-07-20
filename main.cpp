
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;


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

					for (double k = 0; k < 7; k++) {
						double x = (1.0 - (k / 7)) * A.x + (k / 7) * B.x;
						double y = (1.0 - (k / 7)) * A.y + (k / 7) * B.y;
						cv::circle(frame, cv::Point2d(x, y), 2, cv::Scalar(0, 255, 0), -1);
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