#include <iostream>

#include "SparseMatrix.h"

using namespace std;

int main( void ) {
	double s = 19.0; double u = 21.0; double p = 16.0; 
	double e = 5.0;  double r = 18.0; double l = 12.0;
	double i = 1.0; 
	double non_zero_value_A[13] ={19, u, u, l, u, l, p, e, u, l, l, r, i}; 
	int    col_index_A[13] =     { 0, 2, 3, 0, 1, 1, 2, 3, 4, 0, 1, 4, 5}; 
	int    row_pointer_A[6] =    { 0,       3,    5,    7,    9,      13};
	SparseMatrix A(5, 6, non_zero_value_A, col_index_A, row_pointer_A, 13);
	/* [19,  0, 21, 21,  0,  0;
	    12, 21,  0,  0,  0,  0;
	     0, 12, 16,  0,  0,  0;
	     0,  0,  0,  5, 21,  0;
	    12, 12,  0,  0, 18,  1]; */
	cout << A << endl; 

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

	SparseMatrix AmulB = multiple( A, B );
	// Expeted Out put
	// 0    266    0  105  441
	// 252  168    0    0  147
	// 144    0    0    0   84
	// 252  252    0   25  483
	// 360  384    0    0  409
	cout << AmulB << endl; 
	


	//double* B = NULL;
	//double* X = NULL;
	//solve( A, B, X );
}

