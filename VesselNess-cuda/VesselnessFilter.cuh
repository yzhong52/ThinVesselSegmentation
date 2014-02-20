#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Data3D.h"
#include "ImageProcessing.h"
#include <exception>

struct ooops : std::exception {
	string msg;
	ooops(const string& str = "ooops") : msg(str) {}
	const char* what() const { return msg.c_str(); }
};

namespace VesselnessFilterGPU{
	int compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<float>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step, // INPUT 
		float alpha = 1.0e-1f,	// INPUT 
		float beta  = 5.0e0f,	// INPUT 
		float gamma = 3.5e5f ); // INPUT 
}

namespace VFG = VesselnessFilterGPU; 


int VesselnessFilterGPU::compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<float>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step, // INPUT 
		float alpha,	// INPUT parameter 
		float beta,		// INPUT parameter
		float gamma)    // INPUT parameter 
{
#define ST short
#define DT float

	int im_size = dst.get_size_total();

	// allucate memory for the final result
	dst.reset( src.get_size(), 0.0f ); 
	// pointers to temporary memory in the GPU
	short* dev_src = NULL; 
	float* dev_dst = NULL; 
	float* dev_blurred = NULL; 
	float* dev_kernel = NULL; 

	// cuda error message
	cudaError_t cudaStatus;

	// current sigma
	float sigma = sigma_from; 

	int ksize = int(6 * sigma + 1); 
	if( ksize%2==0 ) ksize++; // make sure this is a odd number
	Mat kernel = cv::getGaussianKernel( ksize, sigma, CV_32F );


	try{ 
		const int im_size = src.get_size_total();

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) throw ooops( "Fail to get GPU. " ); 

		// allocate memory for src image
		cudaStatus = cudaMalloc((void**)&dev_src, im_size*sizeof(ST));
		if (cudaStatus != cudaSuccess) throw ooops( "Fail to alloc memory for src image. " ); 
		// copy src image to GPU
		cudaStatus = cudaMemcpy(dev_src, src.getMat().data, im_size*sizeof(ST), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw ooops( "Fail copy src image to GPU. " ); 


		////////////////////////////////////////////////
		// Phrase 1 - Blur the image
		////////////////////////////////////////////////

		// allocate memory for dst
		cudaStatus = cudaMalloc((void**)&dev_dst, im_size*sizeof(DT));
		if (cudaStatus != cudaSuccess) throw ooops( "Fail to alloc memory for dst image. This is also used \
													as a temporary memory when computing Gaussian blur. " );

		// allocate memory for dev_blurred
		cudaStatus = cudaMalloc((void**)&dev_blurred, im_size*sizeof(DT));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 
		
		// allocate memory for kernel
		cudaStatus = cudaMalloc((void**)&dev_kernel, ksize*sizeof(float));
		if (cudaStatus != cudaSuccess) throw cudaStatus; 
		// Copy kernel to GPU
		cudaStatus = cudaMemcpy(dev_kernel, kernel.data, ksize*sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// now blur the image
		// This following magic number varies from computer to computer, 
		// Some better graphic cards support 1024. 
		const int nTPB = 512; 
		// blur image along x direction
		ImageProcessingGPU::cov3<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
			dev_src, dev_blurred, dev_kernel, 
			src.SX(), src.SY(), src.SZ(), 
			ksize, 1, 1 );
		// blur image along y direction
		ImageProcessingGPU::cov3<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
			dev_blurred, dev_dst, dev_kernel, 
			src.SX(), src.SY(), src.SZ(), 
			1, ksize, 1 );
		// blur image along z direction
		ImageProcessingGPU::cov3<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
			dev_dst, dev_blurred, dev_kernel, 
			src.SX(), src.SY(), src.SZ(), 
			1, 1, ksize );




		////////////////////////////////////////////////
		// Phrase 2 - compute the following 
		////////////////////////////////////////////////
		// 1) derivative of the image
		// 2) hessian matrix
		// 3) eigenvalue decomposition of the hessian matrix
		// 4) vesselness measure







		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

		// copy memory from GPU to main memory
		dst.reset( src.get_size() ); 
		cudaStatus = cudaMemcpy(dst.getMat().data, dev_dst, im_size*sizeof(DT), cudaMemcpyDeviceToHost );
		if (cudaStatus != cudaSuccess) throw cudaStatus; 

	} catch( cudaError_t e ) {
		cout << " CUDA Error captured: " << e << ": "; 
		if( e == 11 ) cout << "cudaErrorInvalidValue" << endl; 
	} catch (std::exception& ex) {
		std::cout << "Exception captured: " << ex.what();
	}

	cudaFree(dev_src);
	cudaFree(dev_dst); 
	cudaFree(dev_kernel);
	cudaFree( dev_blurred );

	return 0; 
}
