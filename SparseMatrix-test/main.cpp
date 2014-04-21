#include <iostream>
using namespace std;

#include "../SparseMatrix/SparseMatrix.h"
#ifdef _DEBUG
	#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
	#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif


int main( void ) {
	 
	double non_zero_value_A[13] ={19, 21, 21, 12, 21, 12, 16,  5, 21, 12, 12, 18, 1}; 
	int    col_index_A[13] =     { 0,  2,  3,  0,  1,  1,  2,  3,  4,  0,  1,  4, 5}; 
	int    row_pointer_A[6] =    { 0,          3,      5,      7,      9,        13};
	//SparseMatrix A(5, 6, non_zero_value_A, col_index_A, row_pointer_A, 13);
	///* [19,  0, 21, 21,  0,  0;
	//    12, 21,  0,  0,  0,  0;
	//     0, 12, 16,  0,  0,  0;
	//     0,  0,  0,  5, 21,  0;
	//    12, 12,  0,  0, 18,  1]; */
	//cout << A << endl; 

	///*[ 0, 14,  0,  0,  0;
	//   12,  0,  0,  0,  7;
	//    0,  0,  0,  0,  0;
	//    0,  0,  0,  5, 21;
	//   12, 12,  0,  0, 18;
	//    0,  0,  0,  0,  1;]; */
	//double non_zero_value_B[9] ={14, 12,  7,  5, 21, 12, 12, 18,  1}; 
	//int    col_index_B[9]      ={ 1,  0,  4,  3,  4,  0,  1,  4,  4}; 
	//int    row_pointer_B[6]    ={ 0,  1,    3,3,      5,          8};
	//SparseMatrix B(6, 5, non_zero_value_B, col_index_B, row_pointer_B, 9);
	//cout << B << endl; 

	//SparseMatrix AmulB = multiply( A, B );
	//// Expeted Out put
	//// 0    266    0  105  441
	//// 252  168    0    0  147
	//// 144    0    0    0   84
	//// 252  252    0   25  483
	//// 360  384    0    0  409
	//cout << AmulB << endl; 
	//
	//double non_zero_value_C[13] ={19,-21, 21,-12, 21, 12, 16,  5, 21, 12, 12, 18, 1}; 
	//int    col_index_C[13] =     { 0,  2,  3,  0,  1,  1,  2,  2,  4,  0,  3,  4, 5}; 
	//int    row_pointer_C[6] =    { 0,          3,      5,      7,      9,        13};
	//SparseMatrix C(5, 6, non_zero_value_C, col_index_C, row_pointer_C, 13);
	///* [19,  0, 21, 21,  0,  0;
	//    12, 21,  0,  0,  0,  0;
	//     0, 12, 16,  0,  0,  0;
	//     0,  0,  0,  5, 21,  0;
	//    12, 12,  0,  0, 18,  1]; */
	//cout << A << endl;
	//cout << C << endl;
	//cout << A+C << endl; 
	//cout << A-C << endl; 

	//

	//cout << "=A and A.t()=====================================" << endl; 
	//cout << A << endl;
	//cout << A.t() << endl; 

	//cout << "=A * A.t()=====================================" << endl; 
	///* 1243         228         336         105         228
 //       228         585         252           0         396
 //       336         252         400           0         144
 //       105           0           0         466         378
 //       228         396         144         378         613*/
	//cout << A * (A.t()) << endl; 
	//cout << A.t() * A << endl; 

}

