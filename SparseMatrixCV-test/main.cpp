#ifdef _DEBUG
#pragma comment(lib,"../x64/Debug/SparseMatrixCV.lib")
#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
#pragma comment(lib,"../x64/Release/SparseMatrixCV.lib")
#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif


#include "../SparseMatrixCV/SparseMatrixCV.h"
#include "opencv2\core\core.hpp"
#include <iostream>
using namespace cv; 
using namespace std;



int main( void ) {
	cout << "############################ Identity Matrix ###########################" << endl << endl; 
	cout << SparseMatrixCV::I( 4 ) << endl; 
	
	cout << "############################ Add & Sub #################################" << endl << endl; 
	{
		static const int index[][2] = { {0, 0}, {1, 1}, {2, 2} }; 
		static const double value[] = { 1.0, 1.0, 1.0 }; 
		static const SparseMatrixCV A( 3, 6, index, value, 3 );
		cout << A << endl; 

		static const int indexAA[][2] = { {0, 3}, {1, 4}, {2, 5} }; 
		static const double valueAA[] = { 1.0, 1.0, 1.0 }; 
		static const SparseMatrixCV AA( 3, 6, indexAA, valueAA, 3 );
		cout << AA << endl; 

		cout << A - AA << endl; 
		cout << A + AA << endl; 
	}

	cout << "############################ Constructors   ###########################" << endl << endl; 

	cv::Vec3f v(1.0f, 0.0f, 2.0f);
	static const SparseMatrixCV B( v );
	cout << B << endl; 

	cout << "############################ Multiplications ###########################" << endl << endl; 
	{
		/* [19,  0, 21, 21,  0,  0;
			12, 21,  0,  0,  0,  0;
			 0, 12, 16,  0,  0,  0;
			 0,  0,  0,  5, 21,  0;
			12, 12,  0,  0, 18,  1]; */
		double non_zero_value_A[13] ={19, 21, 21, 12, 21, 12, 16,  5, 21, 12, 12, 18, 1}; 
		int    col_index_A[13] =     { 0,  2,  3,  0,  1,  1,  2,  3,  4,  0,  1,  4, 5}; 
		int    row_pointer_A[6] =    { 0,          3,      5,      7,      9,        13};
		SparseMatrixCV A(5, 6, non_zero_value_A, col_index_A, row_pointer_A, 13);
	
	
		/*[ 0, 14,  0,  0,  0;
		   12,  0,  0,  0,  7;
			0,  0,  0,  0,  0;
			0,  0,  0,  5, 21;
		   12, 12,  0,  0, 18;
			0,  0,  0,  0,  1;]; */
		double non_zero_value_B[9] ={14, 12,  7,  5, 21, 12, 12, 18,  1}; 
		int    col_index_B[9]      ={ 1,  0,  4,  3,  4,  0,  1,  4,  4}; 
		int    row_pointer_B[6]    ={ 0,  1,    3,3,      5,          8};
		SparseMatrixCV B(6, 5, non_zero_value_B, col_index_B, row_pointer_B, 9);
		cout << B << endl; 

		SparseMatrixCV AmulB = A * B;
		// Expeted Out put
		// 0    266    0  105  441
		// 252  168    0    0  147
		// 144    0    0    0   84
		// 252  252    0   25  483
		// 360  384    0    0  409
		cout << AmulB << endl; 


		Mat_<double> DenseB = ( Mat_<double>(6, 5)<< 
			0, 14,  0,  0,  0,
		   12,  0,  0,  0,  7,
			0,  0,  0,  0,  0,
			0,  0,  0,  5, 21,
		   12, 12,  0,  0, 18,
			0,  0,  0,  0,  1 );
		cout << DenseB << endl << endl; 

		Mat_<double> AmulDenseB = A * DenseB; 
		cout << A * DenseB << endl;
	}

	return 0;
}