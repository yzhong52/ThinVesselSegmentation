#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrixDataCol.h"
#include <stdlib.h> //realloc
#include <string.h>
#include <assert.h>

SparseMatrixDataCol::SparseMatrixDataCol( int rows, int cols,
	const double non_zero_value[], const int row_index[], const int col_pointer[], int N )
{
	// non-zero values
	double   *nzval = new double[N];

	memcpy( nzval, non_zero_value, sizeof(double) * N );

	int* rowind = new int[N];
	memcpy( rowind, row_index, sizeof(int) * N );

	int* colptr = new int[rows+1];
	memcpy( colptr, col_pointer, sizeof(int) * cols );
	colptr[cols] = N;
}

SparseMatrixDataCol::SparseMatrixDataCol( int rows, int cols, double nzval[],
	int rowind[], int colptr[], int N )
{
	assert(0);
}


SparseMatrixDataCol::~SparseMatrixDataCol(){

}

