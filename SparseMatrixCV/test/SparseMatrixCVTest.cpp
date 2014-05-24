#include "SparseMatrixCVTest.h"
#include <iostream>
using namespace cv;
using namespace std;

void SparseMatrixCVTest::SetUp()
{
    /* [19,  0, 21, 21,  0,  0;
    	12, 21,  0,  0,  0,  0;
    	 0, 12, 16,  0,  0,  0;
    	 0,  0,  0,  5, 21,  0;
    	12, 12,  0,  0, 18,  1]; */
    double non_zero_value_A1[13] = {19, 21, 21, 12, 21, 12, 16,  5, 21, 12, 12, 18, 1};
    int    col_index_A1[13] =     { 0,  2,  3,  0,  1,  1,  2,  3,  4,  0,  1,  4, 5};
    int    row_pointer_A1[6] =    { 0,          3,      5,      7,      9,        13};
    A1 = SparseMatrixCV(5, 6, non_zero_value_A1, col_index_A1, row_pointer_A1, 13);
    cout << A1 << endl;

    /*[ 0, 14,  0,  0,  0;
       12,  0,  0,  0,  7;
    	0,  0,  0,  0,  0;
    	0,  0,  0,  5, 21;
       12, 12,  0,  0, 18;
    	0,  0,  0,  0,  1;]; */
    double non_zero_value_A2[9] = {14, 12,  7,  5, 21, 12, 12, 18,  1};
    int    col_index_A2[9]      = { 1,  0,  4,  3,  4,  0,  1,  4,  4};
    int    row_pointer_A2[6]    = { 0,  1,    3,3,      5,          8};
    A2 = SparseMatrixCV(6, 5, non_zero_value_A2, col_index_A2, row_pointer_A2, 9);

    const int sizeB = 9;
    double non_zero_value_B[sizeB] = { 2,  1,  3,  2,  6,  8,  6,  8, 18};
    int    col_index_B[sizeB]      = { 0,  1,  2,  0,  1,  2,  0,  1,  2};
    int    row_pointer_B[4]    = { 0,          3,          6,          9};
    SparseMatrixCV B(3, 3, non_zero_value_B, col_index_B, row_pointer_B, sizeB);

    /*[ 1,  0,  2,  0,  0,  0;
        0,  1,  0,  0,  0,  0;
    	0,  0,  1,  0,  0,  0; */
    const int sizeC1 = 4;
    const int indexC1[sizeC1][2] = { {0, 0}, {0, 2}, {1, 1}, {2, 2} };
    const double valueC1[sizeC1] = { 1.0, 2.0, 1.0, 1.0 };
    C1 = SparseMatrixCV ( 3, 6, indexC1, valueC1, sizeC1 );

    /*[ 1,  0,  0,  1,  0,  0;
        0,  0,  0,  0,  1,  0;
    	0,  0,  0,  0,  0,  1; */
    const int sizeC2 = 4;
    const int indexC2[sizeC2][2] = { {0, 0}, {0, 3}, {1, 4}, {2, 5} };
    const double valueC2[sizeC2] = { 1.0, 1.0, 1.0, 1.0 };
    C2 = SparseMatrixCV( 3, 6, indexC2, valueC2, sizeC2 );
}

void SparseMatrixCVTest::TearDown()
{

}

void SparseMatrixCVTest::test_equal( cv::Mat_<double> m0, SparseMatrixCV m1 )
{
    EXPECT_EQ(m0.rows, m1.row() );
    EXPECT_EQ(m0.cols, m1.col() );

    int N = 0;
    const double* nzval = nullptr;
    const int* colidx   = nullptr;
    const int* rowptr   = nullptr;
    m1.getRowMatrixData( N, nzval, colidx, rowptr );

    int vi = 0;
    for( int r=0; r<m1.row(); r++ )
    {
        for( int c=0; c<m1.col(); c++ )
        {
            if( colidx[vi]==c && vi<rowptr[r+1] )
            {
                ASSERT_DOUBLE_EQ(  m0[r][c], nzval[vi++] );
            }
            else
            {
                ASSERT_DOUBLE_EQ(  m0[r][c], 0.0 );
            }
        }
    }
}

void SparseMatrixCVTest::test_equal( cv::Mat_<double> m0, cv::Mat_<double> m1 ){
    EXPECT_EQ(m0.rows, m1.rows );
    EXPECT_EQ(m0.cols, m1.cols );

    for( int r=0; r<m1.rows; r++ ) {
         for( int c=0; c<m1.cols; c++ ) {
            ASSERT_NEAR( m0[r][c], m1[r][c], 1e-5 );
         }
    }
}

Mat_<double> SparseMatrixCVTest::toCvMat( const SparseMatrixCV& m1 )
{
    int N = 0;
    const double* nzval = nullptr;
    const int* colidx   = nullptr;
    const int* rowptr   = nullptr;
    m1.getRowMatrixData( N, nzval, colidx, rowptr );

    Mat_<double> mres =Mat_<double>::zeros( m1.row(), m1.col() );

    int vi = 0;
    for( int r=0; r<m1.row(); r++ )
    {
        for( int c=0; c<m1.col(); c++ )
        {
            if( colidx[vi]==c && vi<rowptr[r+1] )
            {
                mres[r][c] = nzval[vi++];
            }
        }
    }
    return mres;
}
