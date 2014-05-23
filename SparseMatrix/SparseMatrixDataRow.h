#pragma once

// wrapper class for matrix data store in row order
class SparseMatrixDataRow
{
    friend class SparseMatrixData;

public:

    int nnz;	    /* number of nonzeros in the matrix */
    double *nzval;    /* pointer to array of nonzero values, packed by raw */
    int *colind; /* pointer to array of columns indices of the nonzeros */
    int *rowptr; /* pointer to array of beginning of rows in nzval[]
		       and colind[]  */
                    /* Note:
		       Zero-based indexing is used;
		       rowptr[] has nrow+1 entries, the last one pointing
		       beyond the last row, so that rowptr[nrow] = nnz. */



private:

    // constructors & destructors
    SparseMatrixDataRow( int rows, int cols, double nzval[], int colidx[], int rowptr[], int N );
    SparseMatrixDataRow( int rows, int cols, const double non_zero_value[],
                         const int col_index[], const int row_pointer[], int N );
    ~SparseMatrixDataRow();
};
