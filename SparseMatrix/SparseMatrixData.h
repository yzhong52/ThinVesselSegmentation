#pragma once
#include "SparseMatrixDataCol.h"
#include "SparseMatrixDataRow.h"
#include <utility>


#include <string.h>


/*! \brief Convert a row compressed storage into a column compressed storage.
 */



class SparseMatrixData
{
//    struct MatrixData {
//        int nnz;       // number of non-zero value
//        double *nzval; // pointer to array of nonzero values
//        union{
//            // row order representation
//            struct{
//                int *colind; /* pointer to array of columns indices of the nonzeros */
//                int *rowptr; /* pointer to array of beginning of rows in nzval[] and colind[]  */
//            };
//            // column order representtaion
//            struct {
//
//            };
//        };
//    };
public:
    // size of the matrix
    int ncol, nrow;

    SparseMatrixDataCol* datacol; // matrix data stored in collumn order
    SparseMatrixDataRow* datarow; // matrix data stored in row order
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
        return !datacol && !datarow;
    }

    SparseMatrixDataCol* getCol();
    SparseMatrixDataRow* getRow();
    void transpose( void );
    void multiply( const double& value );

private:
    void RowMatrix_to_ColMatrix(int row, int col, int nnz,
                                double *a, int *colind, int *rowptr,
                                double **at, int **rowind, int **colptr);
};

