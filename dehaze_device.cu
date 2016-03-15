#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include "dehaze_device.h"

__global__ void rgbminKernel(unsigned char* input, unsigned int* rgbmin, int rows, int cols, unsigned int* haze);

__global__ void darkChannelKernel(unsigned int* rgbmin, int rows, int cols);

__global__ void reduceKernel(unsigned int* rein, int n);

__global__ void estimateTransmissionKernel(unsigned int* darkChannel, unsigned int* transmission, unsigned int lightA, int rows, int cols);

__global__ void getDehazedKernel(unsigned int* haze, unsigned int* transmission, unsigned char* dehazed, unsigned int lightA, int rows, int cols);

void darkChannelOnDevice(unsigned char* input, unsigned int* rgbmin, int rows, int cols, unsigned int* haze)
{
	dim3 dimBlock(32, 32);
	dim3 dimGrid((cols - 1)/dimBlock.x + 1, (rows-1)/dimBlock.y +1);
	rgbminKernel<<<dimGrid, dimBlock>>>(input, rgbmin, rows, cols, haze);
	cudaThreadSynchronize();
	dim3 dimGrid2((cols-3)/30+1, (rows-3)/30+1);
	darkChannelKernel<<<dimGrid2, dimBlock>>>(rgbmin, rows, cols);
	cudaThreadSynchronize();
}

void estimateAOnDevice(unsigned int* rein, int size)
{
	int n = size;
	while (n>1) {
		reduceKernel<<<(n-1)/1024+1, 1024>>>(rein, n);
		n = (n-1)/1024+1;
	}
	cudaThreadSynchronize();
}

void estimateTransmissionOnDevice(unsigned int* darkChannel, unsigned int* transmission, unsigned int lightA, int rows, int cols)
{
	dim3 dimBlock(32, 32);
	dim3 dimGrid((cols - 1)/dimBlock.x + 1, (rows-1)/dimBlock.y +1);
	estimateTransmissionKernel<<<dimGrid, dimBlock>>>(darkChannel, transmission, lightA, rows, cols);
	cudaThreadSynchronize();
}

void getDehazedOnDevice(unsigned int* haze, unsigned int* transmission, unsigned char* dehazed, unsigned int lightA, int rows, int cols)
{
	dim3 dimBlock(32, 32);
	dim3 dimGrid((cols - 1)/dimBlock.x + 1, (rows-1)/dimBlock.y +1);
	getDehazedKernel<<<dimGrid, dimBlock>>>(haze, transmission, dehazed, lightA, rows, cols);
	cudaThreadSynchronize();
}

__global__ void rgbminKernel(unsigned char* input, unsigned int* rgbmin, int rows, int cols, unsigned int* haze)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
    int row = threadIdx.y;
	int col = threadIdx.x;

	unsigned char intensity0 = 0x00;
	unsigned char intensity1 = 0x00;
	unsigned char intensity2 = 0x00;
	unsigned char pixelmin = 0x00;

	if ((32 * blockRow + row) < rows && (32 * blockCol + col) < cols)
	{
		intensity0 = input[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 0];
		intensity1 = input[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 1];
		intensity2 = input[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 2];
//		rein get data
		unsigned char t0 = intensity0;
		unsigned char t1 = intensity1;
		unsigned char t2 = intensity2;
		haze[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 0] = (unsigned int)t0;
		haze[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 1] = (unsigned int)t1;
		haze[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 2] = (unsigned int)t2;
}
	else
	{
		intensity0 = 0;
		intensity1 = 0;
		intensity2 = 0;
	}
	syncthreads();
	unsigned char tmp = intensity1 < intensity2 ? intensity1 : intensity2;
	pixelmin = intensity0 < tmp ? intensity0 : tmp;

	if ((32 * blockRow + row) < rows && (32 * blockCol + col) < cols)
	{
		rgbmin[(blockRow * 32 + row) * cols + (blockCol * 32 + col)] = (unsigned int)pixelmin;
	}
}

__global__ void darkChannelKernel(unsigned int* rgbmin, int rows, int cols)
{
	__shared__ unsigned int sdata[32][32];
	unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;

	if ((x-2*blockIdx.x)<cols && (y-2*blockIdx.y)<rows) {
		sdata[threadIdx.y][threadIdx.x] = rgbmin[(y-2*blockIdx.y) * cols + (x-2*blockIdx.x)];
	}

	if (threadIdx.x>0 && threadIdx.x< 31 && threadIdx.y>0 && threadIdx.y<31) {
		if ((x-2*blockIdx.x)<(cols-1) && (y-2*blockIdx.y)<(rows-1)) {
			unsigned int window[9];
			window[0] = sdata[threadIdx.y-1][threadIdx.x-1];
			window[1] = sdata[threadIdx.y-1][threadIdx.x];
			window[2] = sdata[threadIdx.y-1][threadIdx.x+1];
			window[3] = sdata[threadIdx.y][threadIdx.x-1];
			window[4] = sdata[threadIdx.y][threadIdx.x];
			window[5] = sdata[threadIdx.y][threadIdx.x+1];
			window[6] = sdata[threadIdx.y+1][threadIdx.x-1];
			window[7] = sdata[threadIdx.y+1][threadIdx.x];
			window[8] = sdata[threadIdx.y+1][threadIdx.x+1];
			syncthreads();
		    // Order elements (only half of them)
		    for (unsigned int j=0; j<5; ++j)
		    {
		        // Find position of minimum element
		        unsigned int min=j;
		        for (unsigned int l=j+1; l<9; ++l)
		            if (window[l] < window[min])
		                min=l;

		        // Put found minimum element in its place
		        unsigned int temp=window[j];
		        window[j]=window[min];
		        window[min]=temp;

		        syncthreads();
		    }
		    rgbmin[(y-2*blockIdx.y) * cols + (x-2*blockIdx.x)] = window[4];
		}
	}
}

__global__ void reduceKernel(unsigned int* rein, int n)
{
	__shared__ unsigned int sdata[1024];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;

	sdata[tid] = 0;

	if (i < n) {
		sdata[tid] = rein[i];
	}
	__syncthreads();

//	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] = sdata[tid]>sdata[tid+512] ? sdata[tid]:sdata[tid+512];
		}
		__syncthreads();
//	}
//	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = sdata[tid]>sdata[tid+256] ? sdata[tid]:sdata[tid+256];
		}
		__syncthreads();
//	}
//	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = sdata[tid]>sdata[tid+128] ? sdata[tid]:sdata[tid+128];
		}
		__syncthreads();
//	}
//	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = sdata[tid]>sdata[tid+64] ? sdata[tid]:sdata[tid+64];
		}
		__syncthreads();
//	}

	if (tid < 32) {
//		if (blockSize >= 64) {
			sdata[tid] = sdata[tid]>sdata[tid+32] ? sdata[tid]:sdata[tid+32];
//		}
//		if (blockSize >= 32) {
			sdata[tid] = sdata[tid]>sdata[tid+16] ? sdata[tid]:sdata[tid+16];
//		}
//		if (blockSize >= 16) {
			sdata[tid] = sdata[tid]>sdata[tid+8] ? sdata[tid]:sdata[tid+8];
//		}
//		if (blockSize >= 8) {
			sdata[tid] = sdata[tid]>sdata[tid+4] ? sdata[tid]:sdata[tid+4];
//		}
//		if (blockSize >= 4) {
			sdata[tid] = sdata[tid]>sdata[tid+2] ? sdata[tid]:sdata[tid+2];
//		}
//		if (blockSize >= 2) {
			sdata[tid] = sdata[tid]>sdata[tid+1] ? sdata[tid]:sdata[tid+1];
//		}
	}

	if (tid == 0) {
		rein[blockIdx.x] = sdata[0];
		__syncthreads();
	}
}

__global__ void estimateTransmissionKernel(unsigned int* darkChannel, unsigned int* transmission, unsigned int lightA, int rows, int cols)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	int row = threadIdx.y;
	int col = threadIdx.x;

	unsigned int dark, trans;

	if ((32 * blockRow + row) < rows && (32 * blockCol + col) < cols){
		dark = darkChannel[(blockRow*32 + row)*cols + blockCol*32 + col];
		trans = (1 - 0.75 * dark/ lightA) * 255;
		transmission[(blockRow*32 + row)*cols + blockCol*32 + col] = trans;
	}
	syncthreads();
}

__global__ void getDehazedKernel(unsigned int* haze, unsigned int* transmission, unsigned char* dehazed, unsigned int lightA, int rows, int cols)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	int row = threadIdx.y;
	int col = threadIdx.x;

	int haze0, haze1, haze2, trans, dehaze0, dehaze1, dehaze2;
	float tmin = 0.1;
	float tmax, td;
	unsigned char pixel0, pixel1, pixel2;

	if ((32 * blockRow + row) < rows && (32 * blockCol + col) < cols){
		haze0 = haze[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 0];
		haze1 = haze[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 1];
		haze2 = haze[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 2];
		trans = transmission[(blockRow*32 + row)*cols + blockCol*32 + col];
		td = (float) trans;
		tmax = (td/255) < tmin ? tmin : (td/255);
		dehaze0 = abs(((int)haze0 - (int)lightA) / tmax + (int)lightA) > 255 ? 255 : abs(((int)haze0 - (int)lightA) / tmax + (int)lightA);
		dehaze1 = abs(((int)haze1 - (int)lightA) / tmax + (int)lightA) > 255 ? 255 : abs(((int)haze1 - (int)lightA) / tmax + (int)lightA);
		dehaze2 = abs(((int)haze2 - (int)lightA) / tmax + (int)lightA) > 255 ? 255 : abs(((int)haze2 - (int)lightA) / tmax + (int)lightA);
		pixel0 = (unsigned char)dehaze0;
		pixel1 = (unsigned char)dehaze1;
		pixel2 = (unsigned char)dehaze2;
		dehazed[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 0] = pixel0;
		dehazed[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 1] = pixel1;
		dehazed[(blockRow * 32 + row) * cols * 3 + 3 * (blockCol * 32 + col) + 2] = pixel2;
	}
	syncthreads();
}

void* AllocateDeviceMemory(size_t size)
{
	void* deviceMemory;
	cudaMalloc((void**)&deviceMemory, size);
	return deviceMemory;
}

void CopyToDevice(void* host, void* device, size_t size)
{
	cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

void CopyFromDevice(void* host, void* device, size_t size)
{
	cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}

void DeviceToDevice(void* dst, void*src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void FreeDeviceMemory(void* deviceMemory)
{
	cudaFree(deviceMemory);
}
