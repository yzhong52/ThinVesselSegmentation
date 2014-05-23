#ifndef SPARSEMATRIXCVTEST_H
#define SPARSEMATRIXCVTEST_H

#include "gtest/gtest.h"
#include "../SparseMatrixCV.h"



class SparseMatrixCVTest : public testing::Test
{
protected:
    SparseMatrixCV A1, A2, B, C1, C2;
    virtual void SetUp();
    virtual void TearDown();

    static void test_equal( cv::Mat_<double> m0, SparseMatrixCV m1 );
    static void test_equal( cv::Mat_<double> m0, cv::Mat_<double> m1 );

    // convert a matrix from SparseMatrixCV to cv::Mat_<double>
    static cv::Mat_<double> toCvMat( const SparseMatrixCV& m1 );
};


#endif // SPARSEMATRIXCVTEST_H
