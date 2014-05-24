#pragma once
#include "SparseMatrixDataCol.h"
#include "SparseMatrixDataRow.h"
#include <utility>
#include <string.h>


#include "MatrixData.h"


class SparseMatrixData
{

public:
    // size of the matrix
    int ncol, nrow;

    MatrixData datacol; // matrix data stored in collumn order
    MatrixData datarow; // matrix data stored in row order
public:
    // create an zero matrix
    SparseMatrixData( unsigned num_rows, unsigned num_cols);

    // constructor & destructor
    // By defaut the is stored as row order
    SparseMatrixData(
        int num_rows,                   // number of row
        int num_cols,                   // number of cols
        const double non_zero_value[],  // non-zero values
        const int col_index[],			// pointer to column indeces
        const int row_pointer[],		// pointers to data of each row
        int N );						// number of non-zero values

    // dtor
    ~SparseMatrixData();

    inline const int& col() const
    {
        return ncol;
    }
    inline const int& row() const
    {
        return nrow;
    }
    inline bool isZero ( void ) const
    {
        return isRow() && isCol();
    }
    inline bool isRow (void) const {
        return datarow.isEmpty();
    }
    inline bool isCol (void) const {
        return datacol.isEmpty();
    }
    void getCol(int& N, const double*& nzval, const int *&rowind, const int*& colptr );
    void getRow(int& N, const double*& nzval, const int *&colind, const int*& rowptr );

    void transpose( void );
    void multiply( const double& value );

private:
    void RowMatrix_to_ColMatrix(int row, int col, int nnz,
                                double *a, int *colind, int *rowptr,
                                double **at, int **rowind, int **colptr);
};

