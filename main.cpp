
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;

int main() {
	cout << "Starting AR Program" << endl;

	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

	cout << "Starting webcam capture..." << endl;
	cv::VideoCapture capture(0);
	cv::Mat frame;
	
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
	
	while (capture.read(frame)) {
		if (cv::waitKey(25) == 27 || !cv::getWindowProperty(outputWindowName, cv::WND_PROP_VISIBLE))
			break;

		cv::imshow("Output", frame);
	}

	cv::destroyAllWindows();
	capture.release();
}