#include <iostream>

#include "SparseMatrix.h"

using namespace std;

int main( void ) {
	SparseMatrix A;
	double* B = NULL;
	double* X = NULL;
	solve( A, B, X );
}

