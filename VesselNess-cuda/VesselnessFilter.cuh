#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Data3D.h"
#include "ImageProcessing.h"
#include "Ooops.h"


namespace VesselnessFilterGPU{
	int compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<float>& dst,       // OUTPUT
		float sigma_from, float sigma_to, float sigma_step, // INPUT 
		float alpha = 1.0e-1f,	  // INPUT - parameters
		float beta  = 5.0e0f,	  // INPUT - parameters 
		float gamma = 3.5e5f );   // INPUT - parameters 


	__global__ void vesselness( float* src, float* dst, 
		int sx, int sy, int sz, 
		float alpha, float beta, float gamma  ); 
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

	// allucate memory for the final result
	dst.reset( src.get_size(), 0.0f ); 
	// pointers to temporary memory in the GPU
	short* dev_src = NULL; 
	float* dev_blurred = NULL; 
	float* dev_kernel = NULL; 
	short* dev_temp = NULL; 
	float* dev_dst = NULL; 

	// cuda error message
	cudaError_t cudaStatus;

	const int im_size = src.get_size_total();
	try{ 
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) throw Ooops( "Fail to get GPU. " ); 
		

		////////////////////////////////////////////////
		// Allocating memory in GPU
		////////////////////////////////////////////////
		// allocate memory for src image
		cudaStatus = cudaMalloc((void**)&dev_src, im_size*sizeof(ST));
		if (cudaStatus != cudaSuccess) throw Ooops( "Fail to alloc memory for src image. " ); 
		// copy src image to GPU
		cudaStatus = cudaMemcpy(dev_src, src.getMat().data, im_size*sizeof(ST), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw Ooops( "Fail copy src image to GPU. " ); 
		// allocate memory for dst
		cudaStatus = cudaMalloc((void**)&dev_dst, im_size*sizeof(DT));
		if (cudaStatus != cudaSuccess) throw Ooops( "Fail to alloc memory for dst image. " );
		// allocate memory for dev_blurred
		cudaStatus = cudaMalloc((void**)&dev_blurred, im_size*sizeof(DT));
		if (cudaStatus != cudaSuccess) throw Ooops( "Fail to alloc memory for blurred image. " );
		// allocate memory for dev_blurred
		cudaStatus = cudaMalloc((void**)&dev_temp, im_size*sizeof(DT));
		if (cudaStatus != cudaSuccess) throw Ooops( "Fail to alloc memory for temp image. " );
		// allocate memory for Gaussian kernel
		int max_ksize = int(6 * sigma_to + 1); // maximum kernel size
		if( max_ksize%2==0 ) max_ksize++; // make sure this is a odd number
		cudaStatus = cudaMalloc((void**)&dev_kernel, max_ksize*sizeof(float));
		if (cudaStatus != cudaSuccess) throw Ooops( "Fail to alloc memory for Gaussian blur kernel. " );

		for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step ){
			int ksize = int( 6*sigma + 1 ); 
			if( ksize%2==0 ) ksize++; 
			Mat kernel = cv::getGaussianKernel( ksize, sigma, CV_32F );

			////////////////////////////////////////////////
			// Phrase 1 - Blur the image
			////////////////////////////////////////////////
			// Copy Gaussian kernel to GPU
			cudaStatus = cudaMemcpy(dev_kernel, kernel.data, ksize*sizeof(float), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) throw cudaStatus; 
			// Blur the image
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
				dev_blurred, dev_temp, dev_kernel, 
				src.SX(), src.SY(), src.SZ(), 
				1, ksize, 1 );
			// blur image along z direction
			ImageProcessingGPU::cov3<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
				dev_temp, dev_blurred, dev_kernel, 
				src.SX(), src.SY(), src.SZ(), 
				1, 1, ksize );

			ImageProcessingGPU::multiply<<<(im_size + nTPB - 1)/nTPB, nTPB>>>(
				dev_blurred, dev_blurred, src.get_size_total(), sigma ); 


			////////////////////////////////////////////////
			// Phrase 2 - compute the following 
			////////////////////////////////////////////////
			// 1) derivative of the image
			// 2) hessian matrix
			// 3) eigenvalue decomposition of the hessian matrix
			// 4) vesselness measure
			VFG::vesselness<<<(im_size + nTPB - 1)/nTPB, nTPB>>>( 
				dev_blurred, dev_dst, 
				src.SX(), src.SY(), src.SZ(), 
				alpha, beta, gamma );

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) throw cudaStatus; 
		}
		
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

	cudaFree( dev_src );
	cudaFree( dev_dst ); 
	cudaFree( dev_kernel );
	cudaFree( dev_temp );
	cudaFree( dev_blurred );

	return 0; 
}


#define get(im, current_index, sisze_x, size_y, size_z, offset_x, offset_y, offset_z) \
	im[(current_index) + (offset_x) + (offset_y)*(sx) + (offset_z)*(sx)*(sy)]


// GPU version of vesselness filter
__global__ void VesselnessFilterGPU::vesselness(
	float* src, // Input Src image
	float* dst, // Output Dst image
	int sx, int sy, int sz, // size of the image
	float alpha, float beta, float gamma ) // parameters for vesselness
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sx*sy*sz ){
		// the image position (ix, iy, iz)
		int ix = i % (sx * sy) % sx; 
		int iy = i % (sx * sy) / sx;
		int iz = i / (sx * sy); 
		
		// we don't compute the vesselness measure for the boarder pixel
		if( ix<=0 || ix>=sx-1 ) return; 
		if( iy<=0 || iy>=sy-1 ) return;
		if( iz<=0 || iz>=sz-1 ) return;

		// The following are being computed in this function
		// 1) derivative of images; 
		// 2) eigenvalue decomposition of Hessian matrix; 
		// 3) vesselness measure. 

		// 1) derivative of the image		
		float im_dx2 = -2.0f * get( src,i,sx,sy,sz, 0, 0, 0 )
			                 + get( src,i,sx,sy,sz,-1, 0, 0 )
						     + get( src,i,sx,sy,sz,+1, 0, 0 ); 
		float im_dy2 = -2.0f * get( src,i,sx,sy,sz, 0, 0, 0 )
			                 + get( src,i,sx,sy,sz, 0,-1, 0 )
						     + get( src,i,sx,sy,sz, 0,+1, 0 ); 
		float im_dz2 = -2.0f * get( src,i,sx,sy,sz, 0, 0, 0 )
			                 + get( src,i,sx,sy,sz, 0, 0,-1 )
						     + get( src,i,sx,sy,sz, 0, 0,+1 ); 
		float im_dxdy =  ( get( src,i,sx,sy,sz,-1,-1, 0 )
			             + get( src,i,sx,sy,sz, 1, 1, 0 )
			             - get( src,i,sx,sy,sz,-1, 1, 0 )
						 - get( src,i,sx,sy,sz, 1,-1, 0 )) * 0.25f; 
		float im_dxdz =  ( get( src,i,sx,sy,sz,-1, 0,-1 )
			             + get( src,i,sx,sy,sz, 1, 0, 1 )
			             - get( src,i,sx,sy,sz,-1, 0, 1 )
						 - get( src,i,sx,sy,sz, 1, 0,-1 )) * 0.25f; 
		float im_dydz =  ( get( src,i,sx,sy,sz, 0,-1,-1 )
			             + get( src,i,sx,sy,sz, 0, 1, 1 )
			             - get( src,i,sx,sy,sz, 0, 1,-1 )
						 - get( src,i,sx,sy,sz, 0,-1, 1 )) * 0.25f;

		// 2) eigenvalue decomposition of Hessian matrix (a real symmetric 3x3 matrix A)
		// http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
		const float& A11 = im_dx2;
		const float& A22 = im_dy2;
		const float& A33 = im_dz2;
		const float& A12 = im_dxdy;
		const float& A13 = im_dxdz;
		const float& A23 = im_dydz;

		float eig1, eig2, eig3;
		float p1 = A12*A12 + A13*A13 + A23*A23;
		if( p1 < 1e-7f ) {
			// A is diagonal.
			eig1 = A11;
			eig2 = A22;
			eig3 = A33;
		}
		else{
			float q = (A11+A22+A33)/3; // trace(A)/3
			float p2 = (A11-q)*(A11-q) + (A22-q)*(A22-q) + (A33-q)*(A33-q) + 2 * p1; 
			float p = sqrt(p2 / 6);

			// B = (1 / p) * (A - q * I), where I is the identity matrix
			float B11 = (1 / p) * (A11-q); 
			float B12 = (1 / p) * (A12-q); float& B21 = B12;
			float B13 = (1 / p) * (A13-q); float& B31 = B13;
			float B22 = (1 / p) * (A22-q); 
			float B23 = (1 / p) * (A23-q); float& B32 = B23;
			float B33 = (1 / p) * (A33-q); 
			// Determinant of a 3 by 3 matrix B
			// http://www.mathworks.com/help/aeroblks/determinantof3x3matrix.html
			float detB = B11*(B22*B33-B23*B32) - B12*(B21*B33-B23*B31) + B13*(B21*B32-B22*B31); 

			// In exact arithmetic for a symmetric matrix  -1 <= r <= 1
			// but computation error can leave it slightly outside this range.
			float r = detB/2;
			float phi; 
			const float M_PI3 = 3.14159265f / 3;
			if( r <= -1.0f ) {
				phi = M_PI3; 
			} else if (r >= 1.0f )
				phi = 0; 
			else {
				phi = acos(r) / 3; 
			}

			// the eigenvalues satisfy eig3 <= eig2 <= eig1
			eig1 = q + 2 * p * cos(phi);
			eig3 = q + 2 * p * cos(phi + 2 * M_PI3);
			eig2 = 3 * q - eig1 - eig3; // % since trace(A) = eig1 + eig2 + eig3
		}

		// so that abs(eig1) < abs(eig2) < abs(eig3)
		if( abs(eig1) > abs(eig2) ) {
			float temp = eig2;
			eig2 = eig1; 
			eig1 = temp;
		}
		if( abs(eig2) > abs(eig3) ) {
			float temp = eig2;
			eig2 = eig3; 
			eig3 = temp;
		}
		if( abs(eig1) > abs(eig2) ) {
			float temp = eig2;
			eig2 = eig1;
			eig1 = temp;
		}

		// 4) vesselness measure
		float vn = 0.0f; 
		// vesselness value
		if( eig2 < 0 && eig3 < 0 ) {
			float lmd1 = abs( eig1 );
			float lmd2 = abs( eig2 );
			float lmd3 = abs( eig3 );
			float A = lmd2 / lmd3;
			float B = lmd1 / sqrt( lmd2*lmd3 );
			float S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
			vn = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
		}
		
		if(vn > dst[i]) dst[i] = vn; 
	}
} 