#ifndef SPARSEMATRIXCVTEST_H
#define SPARSEMATRIXCVTEST_H

#include "gtest/gtest.h"
#include "../SparseMatrix.h"

#include <iostream>

class SparseMatrixTest : public testing::Test
{
protected:
    SparseMatrix A1, A2, B, C1, C2;
    virtual void SetUp();
    virtual void TearDown();

    template<unsigned R, unsigned C>
    void test_equal( double (&expected)[R][C], const SparseMatrix& m );

    template<unsigned R, unsigned C>
    void print( double (&expected)[R][C] );
};

template<unsigned R, unsigned C>
void SparseMatrixTest :: test_equal( double (&expected)[R][C], const SparseMatrix& m )
{
    ASSERT_EQ( R, m.row() );
    ASSERT_EQ( C, m.col() );

    unsigned N = 0;
    const double* nzval = nullptr;
    const unsigned* colidx   = nullptr;
    const unsigned* rowptr   = nullptr;
    m.getRowMatrixData( N, nzval, colidx, rowptr );

    ASSERT_NE( N, 0 );

    unsigned vi = 0;
    for( unsigned r=0; r<m.row(); r++ )
    {
        for( unsigned c=0; c<m.col(); c++ )
        {
            if( colidx[vi]==c && vi<rowptr[r+1] )
            {
                ASSERT_NEAR(expected[r][c], nzval[vi++], 1e-5);
            }
            else
            {
                ASSERT_NEAR(expected[r][c], 0, 1e-5);
            }
        }
    }
}


template<unsigned R, unsigned C>
void SparseMatrixTest :: print( double (&expected)[R][C] )
{
    for( unsigned r=0; r<R; r++ )
    {
        for( unsigned c=0; c<C; c++ )
        {
            std::cout.width(5);
            std::cout << expected[r][c] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#endif // SPARSEMATRIXCVTEST_H
