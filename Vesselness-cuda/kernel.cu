// Header files for opencv
#include <iostream> 
using namespace std;
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv; 

#include "VesselDetector.h"
#include "Data3D.h"
#include "nstdio.h"
#include "VesselNess.h"

int CV_TYPE(const type_info& type){
	if( type==typeid(short) ) {
		return CV_16S;
	} 
	else if ( type==typeid(int) ){
		return CV_32S;
	} 
	else if ( type==typeid(float) ) {
		return CV_32F;
	} 
	else if ( type==typeid(double) ) {
		return CV_64F;
	} 
	else if ( type==typeid(Vesselness) ) {
		return CV_32FC( Vesselness::_size );
	} 
	else if ( type==typeid(Vesselness_Sig) ) {
		return CV_32FC( Vesselness_Sig::_size );
	}
	else if ( type==typeid(Vesselness_Nor) ) {
		return CV_32FC( Vesselness_Nor::_size );
	} 
	else if ( type==typeid(Vesselness_All) ) {
		return CV_32FC( Vesselness_All::_size );
	}
	else if ( type==typeid(unsigned char) ) {
		return CV_8U;
	}
	else if ( type==typeid(unsigned short) ){
		return CV_16U;
	}
	else {
		cout << "Datatype is not supported." << endl;
		return -1;
	}
}


string STR_TYPE(const type_info& type){
	if( type==typeid(short) ) {
		return "short";
	} 
	else if ( type==typeid(int) ){
		return "int";
	} 
	else if ( type==typeid(float) ) {
		return "float";
	} 
	else if ( type==typeid(double) ) {
		return "double";
	} 
	else if( type==typeid(Vesselness) ) {
		stringstream ss;
		ss << "float," << Vesselness::_size;
		return ss.str();
	}
	else if( type==typeid(Vesselness_Sig) ) {
		stringstream ss;
		ss << "float," << Vesselness_Sig::_size;
		return ss.str();
	}
	else if( type==typeid(Vesselness_Nor) ) {
		stringstream ss;
		ss << "float," << Vesselness_Nor::_size;
		return ss.str();
	}
	else if( type==typeid(Vesselness_All) ) {
		stringstream ss;
		ss << "float," << Vesselness_All::_size;
		return ss.str();
	}
	else if( type==typeid(unsigned char) ) {
		return "unsigned_char";
	} 
	else if( type==typeid(unsigned short) ){
		return "unsigned_short";
	}
	else {
		smart_return_value( 0, "Datatype is not supported.", "(*^__^*)Error!");
	}
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	Data3D<short> im_short;
	im_short.load( "../data/data15.data" );
	im_short.show(); 

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, (unsigned int)size>>>(dev_c, dev_a, dev_b);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
