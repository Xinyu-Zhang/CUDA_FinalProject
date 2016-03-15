/*
 * dehaze_device.h
 *
 *  Created on: Mar 4, 2016
 *      Author: Coconut
 */

#ifndef DEHAZE_DEVICE_H_
#define DEHAZE_DEVICE_H_

void darkChannelOnDevice(unsigned char* input, unsigned int* rgbmin, int rows, int cols, unsigned int* haze);

void estimateAOnDevice(unsigned int* rein, int size);

void estimateTransmissionOnDevice(unsigned int* darkChannel, unsigned int* transmission, unsigned int lightA, int rows, int cols);

void getDehazedOnDevice(unsigned int* haze, unsigned int* transmission, unsigned char* dehazed, unsigned int lightA, int rows, int cols);

/* Include below the function headers of any other functions that you implement */

void* AllocateDeviceMemory(size_t size);

void CopyToDevice(void* host, void* device, size_t size);

void CopyFromDevice(void* host, void* device, size_t size);

void DeviceToDevice(void* dst, void*src, size_t size);

void FreeDeviceMemory(void* deviceMemory);

#endif /* DEHAZE_DEVICE_H_ */
