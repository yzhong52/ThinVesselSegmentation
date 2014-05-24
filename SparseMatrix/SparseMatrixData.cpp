#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrixData.h"

#include <iostream>

using namespace std;

SparseMatrixData::SparseMatrixData( unsigned num_rows, unsigned num_cols )
    : ncol( num_cols ), nrow( num_rows )
{

}

SparseMatrixData::SparseMatrixData( unsigned num_rows, unsigned num_cols, const double non_zero_value[],
                                    const unsigned col_index[], const unsigned row_pointer[], unsigned N )
    : ncol( num_cols ), nrow( num_rows )
{
    // N==0, then this is a zero matrix, then don't allocate matrix data
    if( N != 0 )
    {

        datarow.nnz = N;

        // non-zero values
        datarow.nzval = new double[N];
        memcpy( datarow.nzval, non_zero_value, sizeof(double) * N );

        // column index
        datarow.colind = new unsigned[N];
        memcpy( datarow.colind, col_index, sizeof(unsigned) * N );

        // row pointer
        datarow.rowptr = new unsigned[nrow+1];
        memcpy( datarow.rowptr, row_pointer, sizeof(unsigned) * nrow );
        datarow.rowptr[nrow] = N;
    }
}



SparseMatrixData::~SparseMatrixData()
{
    datarow.release();
    datacol.release();
}


void SparseMatrixData::getCol( unsigned& N, const double*& nzval, const unsigned *&rowind, const unsigned*& colptr )
{
    if( !datarow.isEmpty() && datacol.isEmpty() )
    {
        RowMatrix_to_ColMatrix( nrow, ncol,
                                datarow.nnz, datarow.nzval, datarow.colind, datarow.rowptr,
                                datacol.nnz, datacol.nzval, datacol.rowind, datacol.colptr );
    }
    N = datacol.nnz;
    nzval  = datacol.nzval;
    rowind = datacol.rowind;
    colptr = datacol.colptr;
}

void SparseMatrixData::getRow(unsigned& N, const double*& nzval, const unsigned *&colind, const unsigned*& rowptr )
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
            datacol.nnz, datacol.nzval, datacol.rowind, datacol.colptr,
            datarow.nnz, datarow.nzval, datarow.colind, datarow.rowptr );                  // row pointers
    }

    N = datarow.nnz;
    nzval  = datarow.nzval;
    colind = datarow.colind;
    rowptr = datarow.rowptr;
}

void SparseMatrixData::transpose( void )
{
    // tanspose number of rows and columns
    std::swap( nrow, ncol );

    if( this->isCol() && this->isRow() )
    {
        // if the matrix is stored as both row-order and col-order
        std::swap( datarow, datacol );
    }
    else if( this->isRow() )
    {
        datacol = datarow;
        datarow.clear();
    }
    else if( this->isCol() )
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
    for( unsigned i=0; i<datarow.nnz; i++ ) datarow.nzval[i] *= value;
    for( unsigned i=0; i<datacol.nnz; i++ ) datacol.nzval[i] *= value;
}


void SparseMatrixData::RowMatrix_to_ColMatrix(unsigned m, unsigned n,
        const unsigned& nnz1, const double * const nzval1, const unsigned * const colind1, const unsigned * const rowptr1,
        unsigned& nnz2, double *&nzval2, unsigned *&rowind2, unsigned *&colptr2)
{
    nnz2 = nnz1;

    register unsigned i, j;

    /* Allocate storage for another copy of the matrix. */
    nzval2 = new double[nnz1];
    rowind2 = new unsigned[nnz1];
    colptr2 = new unsigned[n+1];
    unsigned *marker = new unsigned[n];
    memset(marker, 0, sizeof(unsigned)*n );

    /* Get counts of each column of A, and set up column pointers */
    for (i = 0; i < m; ++i)
    {
        for (j = rowptr1[i]; j < rowptr1[i+1]; ++j)
        {
            ++marker[colind1[j]];
        }
    }

    colptr2[0] = 0;
    for (j = 0; j < n; ++j)
    {
        colptr2[j+1] = colptr2[j] + marker[j];
        marker[j] = colptr2[j];
    }

    /* Transfer the matrix into the compressed column storage. */
    unsigned col, relpos;
    for (i = 0; i < m; ++i)
    {
        for (j = rowptr1[i]; j < rowptr1[i+1]; ++j)
        {
            col = colind1[j];
            relpos = marker[col];
            rowind2[relpos] = i;
            nzval2[relpos] = nzval1[j];
            ++marker[col];
        }
    }

    delete[] marker;
}
