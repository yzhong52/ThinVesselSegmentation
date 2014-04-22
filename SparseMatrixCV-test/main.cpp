#include <iostream>
#include "../SparseMatrixCV/SparseMatrixCV.h"
using namespace std;

#ifdef _DEBUG
#pragma comment(lib,"../x64/Debug/SparseMatrixCV.lib")
#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
#pragma comment(lib,"../x64/Release/SparseMatrixCV.lib")
#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif

int main( void ) {

	static const int index[][2] = { {0, 0}, {1, 1}, {2, 2} }; 
	static const double value[] = { 1.0, 1.0, 1.0 }; 
	static const SparseMatrixCV A( 3, 6, index, value, 3 );
	cout << A << endl; 

	cv::Vec3f v(1.0f, 0.0f, 2.0f);
	static const SparseMatrixCV B( v );
	cout << B << endl; 

	return 0;
}