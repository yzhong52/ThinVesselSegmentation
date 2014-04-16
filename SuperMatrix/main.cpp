#include <iostream>

#include "SparseMatrix.h"

using namespace std;

int main( void ) {
	double s = 19.0; double u = 21.0; double p = 16.0; 
	double e = 5.0;  double r = 18.0; double l = 12.0;
	double non_zero_values[12] = {s, l, l, u, l, l, u, p, u, e, u, r}; 
	int    row_indeces[12] =     {0, 1, 4, 1, 2, 4, 0, 2, 0, 3, 3, 4}; 
	int    col_pointers[6] =     {0, 3, 6, 8, 10, 12};
	SparseMatrix A(5, 5, non_zero_values, row_indeces, col_pointers, 12);

	double* B = NULL;
	double* X = NULL;
	solve( A, B, X );
}

