#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrixData.h"

#include <iostream>

using namespace std;

SparseMatrixData::SparseMatrixData( unsigned num_rows, unsigned num_cols )
    : ncol( num_cols ), nrow( num_rows )
{

}

SparseMatrixData::SparseMatrixData( int num_rows, int num_cols, const double non_zero_value[],
                                    const int col_index[], const int row_pointer[], int N )
    : ncol( num_cols ), nrow( num_rows )
{
    // N==0, then this is a zero matrix, then don't allocate matrix data
    if( N == 0 ) return;

    datarow.nnz = N;

    // non-zero values
    datarow.nzval = new double[N];
    memcpy( datarow.nzval, non_zero_value, sizeof(double) * N );

    // column index
    datarow.colind = new int[N];
    memcpy( datarow.colind, col_index, sizeof(int) * N );

    // row pointer
    datarow.rowptr = new int[nrow+1];
    memcpy( datarow.rowptr, row_pointer, sizeof(int) * nrow );
    datarow.rowptr[nrow] = N;
}



SparseMatrixData::~SparseMatrixData()
{
}


void SparseMatrixData::getCol( int& N, const double*& nzval, const int *&rowind, const int*& colptr )
{
    if( !datarow.isEmpty() && datacol.isEmpty() )
    {
        RowMatrix_to_ColMatrix( nrow, ncol,
                                datarow.nnz, datarow.nzval,  datarow.colind, datarow.rowptr,
                                &datacol.nzval, &datacol.rowind, &datacol.colptr );
        datacol.nnz = datarow.nnz;
    }
    N = datacol.nnz;
    nzval  = datacol.nzval;
    rowind = datacol.rowind;
    colptr = datacol.colptr;
}

void SparseMatrixData::getRow(int& N, const double*& nzval, const int *&colind, const int*& rowptr )
{
    if( datarow.isEmpty() && !datacol.isEmpty() )
    {
        // IMPORTANT!: ncol and nrow is reversed in the following function.
        // This is because dCompRow_to_CompCol is a function for converting
        // row-order sparse matrix to col-order matrix. But it can also be
        // used to comvert a col matrix into a row one if it is carefully used.
        RowMatrix_to_ColMatrix(
            ncol,						// number of rows
            nrow,						// number of cols
            datacol.nnz,               // number of non-zero entries
            datacol.nzval,             // non-zero entries
            datacol.rowind,            // pretend it is column index array
            datacol.colptr,            // pretend it is row pointers arrays
            &datarow.nzval,                     // non-zero entries (for row-order matrix)
            &datarow.colind,                    // column indeces
            &datarow.rowptr );                  // row pointers
        datarow.nnz = datacol.nnz;
    }

    N = datarow.nnz;
    nzval  = datarow.nzval;
    colind = datarow.colind;
    rowptr = datarow.rowptr;
}
//
//const MatrixData& SparseMatrixData::getCol()
//{
//    if( datarow.nnz==0 || datacol.nnz!=0 ) return datacol;
//
//    RowMatrix_to_ColMatrix(
//        nrow,						// number of rows
//        ncol,						// number of cols
//        datarow.nnz,                // number of non-zero entries
//        datarow.nzval,             // non-zero entries
//        datarow.colind,            // column index
//        datarow.rowptr,            // row pointers
//        &datacol.nzval,                     // non-zero entries (for column-order matrix)
//        &datacol.rowind,                    // row indeces
//        &datacol.colptr );                  // column pointers
//
//    return datacol;
//}
//
//const MatrixData&  SparseMatrixData::getRow()
//{
//    if( datacol.nnz==0 || datarow.nnz!=0 ) return datarow;
//
//    // IMPORTANT!: ncol and nrow is reversed in the following function.
//    // This is because dCompRow_to_CompCol is a function for converting
//    // row-order sparse matrix to col-order matrix. But it can also be
//    // used to comvert a col matrix into a row one if it is carefully used.
//    RowMatrix_to_ColMatrix(
//        ncol,						// number of rows
//        nrow,						// number of cols
//        datacol.nnz,               // number of non-zero entries
//        datacol.nzval,             // non-zero entries
//        datacol.rowind,            // pretend it is column index array
//        datacol.colptr,            // pretend it is row pointers arrays
//        &datarow.nzval,                     // non-zero entries (for row-order matrix)
//        &datarow.colind,                    // column indeces
//        &datarow.rowptr );                  // row pointers
//
//    return datarow;
//}

void SparseMatrixData::transpose( void )
{
    // tanspose number of rows and columns
    std::swap( nrow, ncol );

    if( datarow.nnz!=0 && datacol.nnz!=0 )
    {
        // if the matrix is stored as both row-order and col-order
        std::swap( datarow, datacol );
    }
    else if( datarow.nnz!=0 )
    {
        datacol = datarow;
        datarow.clear();
    }
    else if( datacol.nnz!=0 )
    {
        datarow = datacol;
        datacol.clear();
    }
    else
    {
        // both datacol and datarow are NULL
        // this is a empty matrix, nothing to transpose
    }
}

void SparseMatrixData::multiply( const double& value )
{
    for( int i=0; i<datarow.nnz; i++ ) datarow.nzval[i] *= value;
    for( int i=0; i<datacol.nnz; i++ ) datacol.nzval[i] *= value;
}


void SparseMatrixData::RowMatrix_to_ColMatrix(int m, int n, int nnz,
        double *a, int *colind, int *rowptr,
        double **at, int **rowind, int **colptr)
{
    register int i, j, col, relpos;
    int *marker;

    /* Allocate storage for another copy of the matrix. */
    *at = new double[nnz];
    *rowind = new int[nnz];
    *colptr = new int[n+1]; // (int *) intMalloc(n+1);
    marker = new int[n]; // (int *) intCalloc(n);
    memset(marker, 0, sizeof(int)*n );

    /* Get counts of each column of A, and set up column pointers */
    for (i = 0; i < m; ++i)
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) ++marker[colind[j]];
    (*colptr)[0] = 0;
    for (j = 0; j < n; ++j)
    {
        (*colptr)[j+1] = (*colptr)[j] + marker[j];
        marker[j] = (*colptr)[j];
    }

    /* Transfer the matrix into the compressed column storage. */
    for (i = 0; i < m; ++i)
    {
        for (j = rowptr[i]; j < rowptr[i+1]; ++j)
        {
            col = colind[j];
            relpos = marker[col];
            (*rowind)[relpos] = i;
            (*at)[relpos] = a[j];
            ++marker[col];
        }
    }

    delete[] marker;
}
