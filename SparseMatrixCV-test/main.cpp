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
	SparseMatrixCV A, B;
	SparseMatrixCV A_B = A - B; 
	cout << A_B << endl; 

	cout << "hello world" << endl; 

	system( "pause" ); 
	return 0;
}