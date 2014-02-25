#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Data3D.h"

namespace ImageProcessingGPU{
	template<typename ST, typename DT> 
	cudaError_t GaussianBlur3D( const Data3D<ST>& src, Data3D<DT>& dst, int ksize, double sigma ); 

	template<typename ST, typename DT, typename KT>
	__global__ void cov3( 
		const ST* src, 
		DT* dst, 
		const KT* kernel, 
		int sx, int sy, int sz, 
		int kx, int ky, int kz); 

	template<typename ST, typename DT>
	__global__ void multiply( ST* src, DT* dst, int size, float scale ); 
}

namespace IPG = ImageProcessingGPU; 


template<typename ST, typename DT> 
cudaError_t ImageProcessingGPU::GaussianBlur3D( const Data3D<ST>& src, Data3D<DT>& dst, int ksize, double sigma ){
	smart_return_value( ksize%2!=0, "kernel size should be odd number", cudaErrorApiFailureBase );

	////////////////////////////////////////////////////////////////////////////////////////
	//// Relationship between Sigma and Kernal Size (ksize)
	///////////////////////////////
	//// Option I - Based on OpenCV
	////   Calculate sigma from size: 
	////         sigma = 0.3 * ( (ksize-1)/2 - 1 ) + 0.8 = 0.15*ksize + 0.35
	////   Calculate size from sigma: 
	////         ksize = ( sigma - 0.35 ) / 0.15 = 6.67 * sigma - 2.33
	////   Reference: OpenCv Documentation 2.6.4.0 getGaussianKernel
	////     http://docs.opencv.org/modules/imgproc/doc/filtering.html#creategaussianfilter
	//// Option II - Based on the traditional 99.7%
	////   Calculate size from sigma: 
	////         ksize = 6 * sigma + 1
	////   Calculate sigma from ksize: 
	////         sigma = (size - 1)/6 = 0.17 * size - 0.17

	Mat kernel = cv::getGaussianKernel( ksize, sigma, CV_32F );

	// cuda error message
	cudaError_t cudaStatus;

	ST* dev_src = NULL; 
	DT* dev_dst = NULL; 
	float* dev_kernel = NULL; 

	try{ 
		const int im_size = src.get_size_total();

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// allocate memory for src image
		cudaStatus = cudaMalloc((void**)&dev_src, im_size*sizeof(ST));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 
		// copy src image to GPU
		cudaStatus = cudaMemcpy(dev_src, src.getMat().data, im_size*sizeof(ST), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// allocate memory for dst
		cudaStatus = cudaMalloc((void**)&dev_dst, im_size*sizeof(DT));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 
		
		// allocate memory for kernel
		cudaStatus = cudaMalloc((void**)&dev_kernel, ksize*sizeof(float));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 
		cudaStatus = cudaMemcpy(dev_kernel, kernel.data, ksize*sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// This following magic number varies from computer to computer, 
		// Some better graphic cards support 1024. 
		const int nTPB = 512; 
		// blur image along x direction
		ImageProcessingGPU::cov3<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
			dev_src, dev_dst, dev_kernel, 
			src.SX(), src.SY(), src.SZ(), 
			ksize, 1, 1 );
		// blur image along y direction
		ImageProcessingGPU::cov3<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
			dev_dst, dev_src, dev_kernel, 
			src.SX(), src.SY(), src.SZ(), 
			1, ksize, 1 );
		// blur image along z direction
		ImageProcessingGPU::cov3<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
			dev_src, dev_dst, dev_kernel, 
			src.SX(), src.SY(), src.SZ(), 
			1, 1, ksize );

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// copy memory from GPU to main memory
		dst.reset( src.get_size() ); 
		cudaStatus = cudaMemcpy(dst.getMat().data, dev_dst, im_size*sizeof(DT), cudaMemcpyDeviceToHost );
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

	} catch( cudaError_t e ) {
		cout << " CUDA Error captured: " << e << ": "; 
		if( e == 11 ) cout << "cudaErrorInvalidValue" << endl; 
	}

	cudaFree(dev_src);
	cudaFree(dev_dst); 
	cudaFree(dev_kernel);

	return cudaStatus;
}

template<typename ST, typename DT, typename KT>
__global__ void ImageProcessingGPU::cov3(
	const ST* src, // Input Src image
	DT* dst, // Output Dst image
	const KT* kernel, // convolution kerkel
	int sx, int sy, int sz, // size of the image 
	int kx, int ky, int kz) // size of the convolution kernel
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sx*sy*sz ){
		// the image position (ix, iy, iz)
		int ix = i % (sx * sy) % sx; 
		int iy = i % (sx * sy) / sx;
		int iz = i / (sx * sy); 
		// now blur the image
		dst[i] = 0; 
		float sum = 0; 
		// relative position of the kernel (x, y, z)
		for( int x=-kx/2; x<=kx/2; x++ ) {
			for( int y=-ky/2; y<=ky/2; y++ ){
				for( int z=-kz/2; z<=kz/2; z++ ) {
					// if the image position is out of range, continue
					if( ix+x<0 || ix+x>=sx ) continue; 
					if( iy+y<0 || iy+y>=sy ) continue; 
					if( iz+z<0 || iz+z>=sz ) continue; 
					// get the index of the image
					int index = i + x + y*sx + z*sx*sy; 
					// get the index of the kernel
					int index2 = (x+kx/2) + (y+ky/2)*kx + (z+kz/2)*kx*ky; 
					// convolution
					dst[i] += src[index] * kernel[index2];
					sum += kernel[index2]; 
				}
			}
		}
		dst[i] /= sum;
	}
} 

template<typename ST, typename DT>
__global__ void ImageProcessingGPU::multiply( ST* src, DT* dst, int size, float scale ){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size ){
		dst[i] = src[i] * scale;
	}
}