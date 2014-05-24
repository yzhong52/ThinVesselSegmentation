#pragma once
#include <utility>
#include <string.h>

class SparseMatrixData
{
    struct MatrixData
    {
        unsigned nnz;       // number of non-zero value
        double *nzval; // pointer to array of nonzero values
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

        inline void clear()
        {
            nnz = 0;
            nzval = nullptr;
            colind = nullptr;
            rowptr = nullptr;
        }

        bool isEmpty() const
        {
            return (nnz==0);
        }
    };
public:
    // size of the matrix
    unsigned ncol, nrow;

    MatrixData datacol; // matrix data stored in collumn order
    MatrixData datarow; // matrix data stored in row order
public:
    // create an zero matrix
    SparseMatrixData( unsigned num_rows, unsigned num_cols);

    // constructor & destructor
    // By defaut the is stored as row order
    SparseMatrixData(
        unsigned num_rows,                   // number of row
        unsigned num_cols,                   // number of cols
        const double non_zero_value[],  // non-zero values
        const unsigned col_index[],			// pointer to column indeces
        const unsigned row_pointer[],		// pointers to data of each row
        unsigned N );						// number of non-zero values

    // dtor
    ~SparseMatrixData();

    inline const unsigned& col() const
    {
        return ncol;
    }
    inline const unsigned& row() const
    {
        return nrow;
    }
    inline bool isZero ( void ) const
    {
        return !isRow() && !isCol();
    }
    inline bool isRow (void) const
    {
        return !datarow.isEmpty();
    }
    inline bool isCol (void) const
    {
        return !datacol.isEmpty();
    }

    // getter of the matrix in either column representation or row representation
    void getCol(unsigned& N, const double*& nzval, const unsigned *&rowind, const unsigned*& colptr );
    void getRow(unsigned& N, const double*& nzval, const unsigned *&colind, const unsigned*& rowptr );

    void transpose( void );
    void multiply( const double& value );

private:
    void RowMatrix_to_ColMatrix(unsigned m, unsigned n,
        const unsigned& nnz1, const double * const nzval1, const unsigned * const colind1, const unsigned * const rowptr1,
        unsigned& nnz2, double *&nzval2, unsigned *&rowind2, unsigned *&colptr2);
};

