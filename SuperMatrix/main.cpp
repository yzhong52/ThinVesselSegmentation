#include <iostream>

#include "SparseMatrix.h"

using namespace std;

int main( void ) {
	double s = 19.0; double u = 21.0; double p = 16.0; 
	double e = 5.0;  double r = 18.0; double l = 12.0;
	double non_zero_values[12] = {s, u, u, l, u, l, p, e, u, l, l, r}; 
	int    col_indeces[12] =     {0, 2, 3, 0, 1, 1, 2, 3, 4, 0, 1, 4}; 
	int    row_pointers[6] =     {0,       3,    5,    7,    9};
	SparseMatrix A(5, 5, non_zero_values, col_indeces, row_pointers, 12);

	cout << A << endl;

	//double* B = NULL;
	//double* X = NULL;
	//solve( A, B, X );
}

