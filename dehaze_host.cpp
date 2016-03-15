#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "dehaze_device.h"

using namespace cv;

int main(int argc, char** argv)
{
	// Read image;
    Mat haze;
    if (argc == 2) {
        haze = imread(argv[1], CV_LOAD_IMAGE_COLOR);
        namedWindow("haze image", CV_WINDOW_AUTOSIZE);
        imshow("haze image", haze);
    } else {
    	haze = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
		namedWindow("haze image", CV_WINDOW_AUTOSIZE);
		imshow("haze image", haze);
    }

    int rows = haze.rows;
    int cols = haze.cols;

    Mat hazeRemoved = Mat::zeros(haze.rows, haze.cols, CV_8UC3);
    Mat ref = Mat::zeros(haze.rows, haze.cols, CV_8UC3);
    Mat ref2 = Mat::zeros(haze.rows, haze.cols, CV_8UC1);

    // Initial memory;
    uchar* input = (uchar*)AllocateDeviceMemory(rows*cols*3*sizeof(uchar));
    unsigned int* rgbmin = (unsigned int*)AllocateDeviceMemory(rows*cols*sizeof(unsigned int));
    unsigned int* original = (unsigned int*)AllocateDeviceMemory(rows*cols*3*sizeof(unsigned int));
    unsigned int* rein = (unsigned int*)AllocateDeviceMemory(rows*cols*sizeof(unsigned int));
    unsigned int* transmission = (unsigned int*)AllocateDeviceMemory(rows*cols*sizeof(unsigned int));
    uchar* dehazed = (uchar*)AllocateDeviceMemory(rows*cols*3*sizeof(uchar));
    unsigned int* read = (unsigned int*)malloc(rows*cols*sizeof(unsigned int));
    unsigned int* A = (unsigned int*)malloc(1*sizeof(unsigned int));

    CopyToDevice(haze.data, input, rows*cols*3*sizeof(uchar));
    // Calculate dark channel;
    darkChannelOnDevice(input, rgbmin, rows, cols, original);
    DeviceToDevice(rein, rgbmin, rows*cols*sizeof(unsigned int));
    // Calculate light A;
    estimateAOnDevice(rein, rows*cols);
    CopyFromDevice(A, rein, 1*sizeof(unsigned int));
    unsigned int lightA = A[0];
    std::cout<<"The light A of the image is: "<<lightA<<std::endl;
    // Calculate transmission;
    estimateTransmissionOnDevice(rgbmin, transmission, lightA, rows, cols);
    // Get haze free image;
    getDehazedOnDevice(original, transmission, dehazed, lightA, rows, cols);

    CopyFromDevice(hazeRemoved.data, dehazed, rows*cols*3*sizeof(uchar));
    // Show and save dark channel;
    CopyFromDevice(read, rgbmin, rows*cols*sizeof(unsigned int));
    for (int i = 0; i<rows*cols; i++){
    	ref2.data[i] = (unsigned char)read[i];
    }
    ref = ref2;
    imshow("darkchannel", ref);
    imwrite("./darkchannel.jpg", ref);
    // Show and save transmission;
    CopyFromDevice(read, transmission, rows*cols*3*sizeof(unsigned int));
	for (int i = 0; i<rows*cols; i++){
		ref2.data[i] = (unsigned char)read[i];
	}
	ref = ref2;
	imshow("transmission", ref);
	imwrite("./transmission.jpg", ref);

    namedWindow("dehaze image", CV_WINDOW_AUTOSIZE);
    imwrite("./dehazed.jpg", hazeRemoved);
    imshow("dehaze image", hazeRemoved);
    waitKey();
    // Free memory;
    FreeDeviceMemory(input);
    FreeDeviceMemory(rgbmin);
    FreeDeviceMemory(original);
    FreeDeviceMemory(rein);
    FreeDeviceMemory(transmission);
    FreeDeviceMemory(dehazed);
    free(A);
    free(read);
    return 0;
}
