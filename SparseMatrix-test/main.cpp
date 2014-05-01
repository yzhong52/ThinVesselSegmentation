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
	SparseMatrix A(5, 6, non_zero_value_A, col_index_A, row_pointer_A, 13);
	/* [19,  0, 21, 21,  0,  0;
	    12, 21,  0,  0,  0,  0;
	     0, 12, 16,  0,  0,  0;
	     0,  0,  0,  5, 21,  0;
	    12, 12,  0,  0, 18,  1]; */
	cout << "############################ Constructor A  ############################" << endl << endl; 
	cout << A << endl; 
	A.print( cout ); 
	return 0; 
	

	cout << "############################ Transpose A.t() ############################" << endl << endl; 
	if( false ) 
	{
	
		{
			double non_zero_value_A[6] = { 1,  1,  1,  1,  1,  1 }; 
			int    col_index_A[6] =      { 0,  1,  2,  1,  2,  2 }; 
			int    row_pointer_A[6] =    { 0,          3,      5, 6, 6,6};
			SparseMatrix A( 5, 6, non_zero_value_A, col_index_A, row_pointer_A, 6 );
			cout << A << endl; 
			cout << A.t() << endl; 
		}
		{
			const int N = 1; 
			const int rows = 8;
			const int cols = 4; 
			double non_zero_value_A[1] = { 1 }; 
			int    col_index_A[1] =      { 3 }; 
			int    row_pointer_A[rows+1] =    { 0,0,1,1,1,1,1,1,1};
			SparseMatrix A( rows, cols, non_zero_value_A, col_index_A, row_pointer_A, 1 );
			cout << A << endl; 
			cout << A.t() << endl; 
		}
		{
			const int N = 2; 
			const int rows = 1;
			const int cols = 12; 
			double non_zero_value_A[N] = { 1, 1 }; 
			int    col_index_A[N] =      { 0, 1 }; 
			int    row_pointer_A[rows+1] =    { 0, 2 };
			SparseMatrix A( rows, cols, non_zero_value_A, col_index_A, row_pointer_A, N );
			cout << A << endl; 
			cout << A.t() << endl; 
		}
		return 0; 
	}

	cout << "############################ A * 2 and A*=2 ############################" << endl << endl; 
	cout << A * 2 << endl; 
	cout << (A *= 2) << endl; 

	/*[ 0, 14,  0,  0,  0;
	   12,  0,  0,  0,  7;
	    0,  0,  0,  0,  0;
	    0,  0,  0,  5, 21;
	   12, 12,  0,  0, 18;
	    0,  0,  0,  0,  1;]; */
	double non_zero_value_B[9] ={14, 12,  7,  5, 21, 12, 12, 18,  1}; 
	int    col_index_B[9]      ={ 1,  0,  4,  3,  4,  0,  1,  4,  4}; 
	int    row_pointer_B[6]    ={ 0,  1,    3,3,      5,          8};
	SparseMatrix B(6, 5, non_zero_value_B, col_index_B, row_pointer_B, 9);
	cout << B << endl; 

	SparseMatrix AmulB = A * B;
	// Expeted Out put
	// 0    266    0  105  441
	// 252  168    0    0  147
	// 144    0    0    0   84
	// 252  252    0   25  483
	// 360  384    0    0  409
	cout << AmulB << endl; 
	
	double non_zero_value_C[13] ={19,-21, 21,-12, 21, 12, 16,  5, 21, 12, 12, 18, 1}; 
	int    col_index_C[13] =     { 0,  2,  3,  0,  1,  1,  2,  2,  4,  0,  3,  4, 5}; 
	int    row_pointer_C[6] =    { 0,          3,      5,      7,      9,        13};
	SparseMatrix C(5, 6, non_zero_value_C, col_index_C, row_pointer_C, 13);
	/* [19,  0, 21, 21,  0,  0;
	    12, 21,  0,  0,  0,  0;
	     0, 12, 16,  0,  0,  0;
	     0,  0,  0,  5, 21,  0;
	    12, 12,  0,  0, 18,  1]; */
	cout << A << endl;
	cout << C << endl;
	cout << A+C << endl; 
	cout << A-C << endl; 

	cout << "==A and A.t()=====================================" << endl; 
	cout << A << endl;
	cout << A.t() << endl; 

	cout << "==A * A.t()=====================================" << endl; 
	/* 1243         228         336         105         228
		228         585         252           0         396
		336         252         400           0         144
		105           0           0         466         378
		228         396         144         378         613*/
	cout << A * A.t() << endl; 
	cout << A.t() * A << endl; 

	cout << (A.t() * A).diag() << endl; 
	cout << (A * A.t()).diag() << endl; 
	return 0; 
}

