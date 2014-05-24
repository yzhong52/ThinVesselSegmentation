#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrixDataRow.h"
#include <string.h>
#include <assert.h>


#include <iostream>

using namespace std;

SparseMatrixDataRow::SparseMatrixDataRow( int rows, int cols,
        const double non_zero_value[], const int col_index[], const int row_pointer[], int N )
{
    this->nnz = N;

    // non-zero values
    this->nzval = new double[N];
    memcpy( nzval, non_zero_value, sizeof(double) * N );

    this->colind = new int[N];
    // if ( !(colind = intMalloc(N)) ) ABORT("Fail to alloc memory for SparseMatrix");
    memcpy( colind, col_index, sizeof(int) * N );

    this->rowptr = new int[rows+1];
    // if ( !(rowptr = intMalloc(rows+1)) ) ABORT("Fail to alloc memory for SparseMatrix");
    memcpy( rowptr, row_pointer, sizeof(int) * rows );
    rowptr[rows] = N;
}

SparseMatrixDataRow::~SparseMatrixDataRow()
{

}
