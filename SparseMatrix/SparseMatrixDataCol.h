#pragma once

// wrapper class for matrix data store in col order
class SparseMatrixDataCol
{
public:
    int nnz;	    /* number of nonzeros in the matrix */
    double *nzval;    /* pointer to array of nonzero values, packed by column */
    int *rowind; /* pointer to array of row indices of the nonzeros */
    int *colptr; /* pointer to array of beginning of columns in nzval[], and rowind[]  */
    /* Note:
    Zero-based indexing is used;
    colptr[] has ncol+1 entries, the last one pointing
    beyond the last column, so that colptr[ncol] = nnz. */

    // constructors & destructors
    //SparseMatrixDataCol( int rows, int cols, const double non_zero_value[],
    //                     const int row_index[], const int col_pointer[], int N );
    ~SparseMatrixDataCol();
};
