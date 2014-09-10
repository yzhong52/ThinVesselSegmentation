#pragma once

#include "SparseMatrixData.h"
#include "RC.h"
#include <vector>
#include <iostream>
#include <ostream>

// This SparseMatrix implementation, once the matrix is created
// it is almost impossible to modify the values anymore.
// but it is designe for solving linear equations.

class SparseMatrix
{
protected:
    // matrix data are all store in class SparseMatrixData
    SparseMatrixData *data;
    // for reference counting
    RC *rc;

    // if this is a zero matrix
    inline bool isZero(void) const
    {
        return data->isZero();
    };
public:
    // c'tor: initialized a zero matrix
    SparseMatrix( unsigned num_rows = 0, unsigned num_cols = 0 );

    // c'tor: initialize a matrix with N non-zero values
    SparseMatrix( unsigned num_rows, unsigned num_cols,
                  const double non_zero_value[],
                  const unsigned col_index[],
                  const unsigned row_pointer[],
                  unsigned N );

    // c'tor
    SparseMatrix( unsigned num_rows, unsigned num_cols,
                  const std::vector<double>&   non_zero_value,
                  const std::vector<unsigned>& col_index,
                  const std::vector<unsigned>& row_pointer );

    // c'tor: copy constructor
    SparseMatrix( const SparseMatrix& matrix );
    const SparseMatrix& operator=( const SparseMatrix& matrix );

    // deep copy of the matrix data
    const SparseMatrix clone(void) const;

    // destructor
    ~SparseMatrix(void);

    // get the number of rows and column of the matrix
    inline const unsigned& row() const
    {
        return data->row();
    }
    inline const unsigned& col() const
    {
        return data->col();
    }

    void getRowMatrixData( unsigned& N, double const*& non_zero_value, unsigned const*& column_index,
                           unsigned const*& row_pointer ) const;

protected:

    bool updateData( unsigned num_rows, unsigned num_cols,
                     const std::vector<double>& non_zero_value,
                     const std::vector<unsigned>& col_index,
                     const std::vector<unsigned>& row_pointer );
    bool updateData(  unsigned num_rows, unsigned num_cols,
                      const double non_zero_value[],
                      const unsigned col_index[],
                      const unsigned row_pointer[],
                      unsigned N );

public:
    ////////////////////////////////////////////////////////////////
    // Matrix manipulations
    ////////////////////////////////////////////////////////////////

    // Transpose a matrix
    const SparseMatrix t() const;
    // mutiply by value
    const SparseMatrix& operator*=( const double& value );
    const SparseMatrix& operator/=( const double& value );
    // solving linear system
    friend void solve( const SparseMatrix& A, const double* B, double* X );
    // other matrix manipulations
    friend const SparseMatrix operator*( const SparseMatrix& m1, const SparseMatrix& m2 );
    friend const SparseMatrix operator+( const SparseMatrix& m1, const SparseMatrix& m2 );
    friend const SparseMatrix operator-( const SparseMatrix& m1, const SparseMatrix& m2 );
    friend const SparseMatrix operator/( const SparseMatrix& m1, const double& value );
    friend const SparseMatrix operator*( const SparseMatrix& m1, const double& value );

    // parallel function(s)
    friend const SparseMatrix multiply_openmp( const SparseMatrix& m1, const SparseMatrix& m2 );

    // utility functions
    void print( std::ostream& out ) const;
    friend std::ostream& operator<<( std::ostream& out, const SparseMatrix& m );

    // return the diaganal matrix
    SparseMatrix diag() const;

    /////////////////////////////////////////////////////////////////
    // friends function for liner sover
    // // // // // // // // // // // // // // // // // // // // // //
    enum Options { BICGSQ, SUPERLU };
    friend void mult( const SparseMatrix& A, const double *v, double *w );
    friend void solve( const SparseMatrix& A, const double* B, double* X,
                       double acuracy = 1e-3, Options o = BICGSQ );
};
