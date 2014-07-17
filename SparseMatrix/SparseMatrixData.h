#pragma once
#include <utility>
#include <string.h>

class SparseMatrixData
{

    struct MatrixData
    {
        unsigned nnz;       // number of non-zero value
        double *nzval;      // pointer to array of nonzero values

        // The data could either be stored in row-major order or column-major order
        union
        {
            // row order representation
            struct
            {
                unsigned *colind; // pointer to array of columns indices of the nonzeros
                unsigned *rowptr; // pointer to array of beginning of rows in nzval[] and colind[]
            };
            // column order representtaion
            struct
            {
                unsigned *rowind; // pointer to array of row indices of the nonzeros
                unsigned *colptr; // pointer to array of beginning of columns in nzval[], and rowind[]
            };
        };

        MatrixData()
            : nnz(0), nzval(nullptr), colind(nullptr), rowptr(nullptr) {}

        inline void release()
        {
            delete[] nzval;
            delete[] colind;
            delete[] rowptr;
            clear();
        }

        // A sparse matrix is 'empty' if there is no non zero values in the matrix
        inline bool isEmpty() const
        {
            return (nnz==0);
        }

        inline void clear()
        {
            nnz = 0;
            nzval = nullptr;
            colind = nullptr;
            rowptr = nullptr;
        }
    };
public:

    // size of the matrix - number of rows and number of columns
    unsigned ncol, nrow;

    MatrixData datacol; // matrix data stored in collumn-major order
    MatrixData datarow; // matrix data stored in row-major order

public:
    // create an zero matrix
    SparseMatrixData( unsigned num_rows, unsigned num_cols);

    // constructor - by defaut the is stored as row order
    SparseMatrixData(
        unsigned num_rows,                   // number of row
        unsigned num_cols,                   // number of cols
        const double non_zero_value[],       // non-zero values
        const unsigned col_index[],			 // pointer to column indeces
        const unsigned row_pointer[],		 // pointers to data of each row
        unsigned N );						 // number of non-zero values

    // destructor
    ~SparseMatrixData();

    inline const unsigned& col() const
    {
        return ncol;
    }
    inline const unsigned& row() const
    {
        return nrow;
    }

    // The matrix is a zero matrix
    inline bool isZero ( void ) const
    {
        return !isRow() && !isCol();
    }

    // The matrix has a row-major representation
    inline bool isRow (void) const
    {
        return !datarow.isEmpty();
    }

    // The matrix has a column-major representation
    inline bool isCol (void) const
    {
        return !datacol.isEmpty();
    }

    // getter of the matrix in either column representation or row representation
    void getCol(unsigned& N, const double*& nzval, const unsigned *&rowind, const unsigned*& colptr );
    void getRow(unsigned& N, const double*& nzval, const unsigned *&colind, const unsigned*& rowptr );

    // tranpose a matrix
    void transpose( void );

    // multiple the matrix by a value
    void multiply( const double& value );

private:
    void RowMatrix_to_ColMatrix(unsigned m, unsigned n,
                                const unsigned& nnz1, const double * const nzval1, const unsigned * const colind1, const unsigned * const rowptr1,
                                unsigned& nnz2, double *&nzval2, unsigned *&rowind2, unsigned *&colptr2);
};

