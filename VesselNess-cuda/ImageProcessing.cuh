#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ImageProcessingGPU{
	template<typename ST, typename DT> 
	cudaError_t GaussianBlur3D( const Data3D<ST>& src, Data3D<DT>& dst, int ksize, double sigma ); 

	template<typename ST, typename DT, typename KT>
	__global__ void cov3( ST* src, DT* dst, KT* kernel, 
		int sx, int sy, int sz, 
		int kx, int ky, int kz); 



	template<typename ST, typename DT, typename KT>
	__global__ void addKernel(ST *c, const DT *a, const KT *b)
	{
		int i = threadIdx.x;
		c[i] = a[i] + b[i];
	}

	// Helper function for using CUDA to add vectors in parallel.
	template<typename ST, typename DT>
	cudaError_t addWithCuda(DT *dst, const ST *src1, const ST *src2, size_t size)
	{
		DT *dev_dst = 0;
		ST *dev_src1 = 0;
		ST *dev_src2 = 0;
		cudaError_t cudaStatus;

		try{
			// Choose which GPU to run on, change this on a multi-GPU system.
			cudaStatus = cudaSetDevice(0);
			if (cudaStatus != cudaSuccess) throw cudaStatus;

			// Allocate GPU buffers for three vectors (two input, one output)    .
			cudaStatus = cudaMalloc((void**)&dev_dst, size * sizeof(DT));
			if (cudaStatus != cudaSuccess) throw cudaStatus;

			cudaStatus = cudaMalloc((void**)&dev_src1, size * sizeof(ST));
			if (cudaStatus != cudaSuccess) throw cudaStatus;

			cudaStatus = cudaMalloc((void**)&dev_src2, size * sizeof(ST));
			if (cudaStatus != cudaSuccess) throw cudaStatus;

			// Copy input vectors from host memory to GPU buffers.
			cudaStatus = cudaMemcpy(dev_src1, src1, size * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) throw cudaStatus;

			cudaStatus = cudaMemcpy(dev_src2, src2, size * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) throw cudaStatus;

			// Launch a kernel on the GPU with one thread for each element.
			addKernel<<<1, (unsigned int)size>>>(dev_dst, dev_src1, dev_src2);

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) throw cudaStatus;

			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(dst, dev_dst, size * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) throw cudaStatus;
		}
		catch( ... ) {

		}

		cudaFree(dev_dst);
		cudaFree(dev_src1);
		cudaFree(dev_src2);

		return cudaStatus;
	}

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

	Mat kernel = cv::getGaussianKernel( ksize, sigma, CV_64F );

	// cuda error message
	cudaError_t cudaStatus;

	ST* dev_src = NULL; 
	DT* dev_dst = NULL; 
	double* dev_kernel = NULL; 

	const int arraySize = src.get_size_total();
    
	try{ 
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// allocate memory for src
		//cudaStatus = cudaMalloc((void**)&dev_src, src.get_size_total() * sizeof(ST));
		//if (cudaStatus != cudaSuccess) throw cudaStatus; 
		//cudaStatus = cudaMemcpy(dev_src, src.getMat().data, src.get_size_total() * sizeof(ST), cudaMemcpyHostToDevice);
		//if (cudaStatus != cudaSuccess) throw cudaStatus; 
		cudaStatus = cudaMalloc((void**)&dev_src, arraySize*sizeof(ST));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 
		cudaStatus = cudaMemcpy(dev_src, src.getMat().data, arraySize*sizeof(ST), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw cudaStatus; 
		
		// allocate memory for dst
		cudaStatus = cudaMalloc((void**)&dev_dst, arraySize*sizeof(DT));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// allocate memory for kernel
		cudaStatus = cudaMalloc((void**)&dev_kernel, arraySize*sizeof(double));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// This number vary from computer to computer. Some better graphic cards support 1024. 
		const int nTPB = 512; 
		ImageProcessingGPU::cov3<<<(arraySize + nTPB - 1)/nTPB, nTPB>>>(
			dev_src, dev_dst, dev_kernel, 
			src.SX(), src.SY(), src.SZ(), 
			ksize, 0, 0 );

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		dst.reset( src.get_size() ); 
		//cudaStatus = cudaMemcpy(dst.getMat().data, dev_dst, dst.get_size_total() * sizeof(DT), cudaMemcpyDeviceToHost );
		//if (cudaStatus != cudaSuccess) throw cudaStatus; 
		cudaStatus = cudaMemcpy(dst.getMat().data, dev_dst, arraySize*sizeof(int), cudaMemcpyDeviceToHost );
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		for( int i=0; i<5; i++ ) cout << dst.at(i,0,0) << " ";
		cout << endl;

	} catch( cudaError_t e ) {
		cout << " CUDA Error captured: " << e << endl; 
		if( e == 11 ) cout << "cudaErrorInvalidValue" << endl; 
	}

	cudaFree(dev_src);
	cudaFree(dev_dst); 
	cudaFree(dev_kernel);

	return cudaStatus;
}


template<typename ST, typename DT, typename KT>
__global__ void ImageProcessingGPU::cov3( ST* src, DT* dst, KT* kernel, 
	int sx, int sy, int sz, int kx, int ky, int kz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sx*sy*sz ) dst[i] = src[i];
} 