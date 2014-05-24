#include "gtest/gtest.h"

#include "SparseMatrixCVTest.h"

using namespace std;
using namespace cv;

// Tests matrix multiplication
TEST_F(SparseMatrixCVTest, Multiplication)
{
    cout << "A1 = " << A1 << endl;
    cout << "A2 = " << A2 << endl;

    cout << "Expected resutl: " << endl;
    cout << "toCvMat(A1) * toCvMat(A2) = " << toCvMat(A1) * toCvMat(A2) << endl;

    cout << "A1 * A2 = " << A1 * A2 << endl;
    test_equal( toCvMat(A1) * toCvMat(A2), A1 * A2);
}


TEST_F(SparseMatrixCVTest, Subtraction)
{
    test_equal( toCvMat( C1 ) - toCvMat( C2 ), C1 - C2 );
}


TEST_F(SparseMatrixCVTest, Addition)
{
    test_equal( toCvMat( C1 ) + toCvMat( C2 ), C1 + C2 );
}


// Tests matrix inverse
TEST_F(SparseMatrixCVTest, Inverse)
{
    double non_zero_value_A[9] = { 2,  1,  3,  2,  6,  8,  6,  8, 18};
    unsigned    col_index_A[9]      = { 0,  1,  2,  0,  1,  2,  0,  1,  2};
    unsigned    row_pointer_A[4]    = { 0,          3,          6,          9};
    SparseMatrixCV A(3, 3, non_zero_value_A, col_index_A, row_pointer_A, 9);

    Mat_<double> B = (Mat_<double>(3,1) << 1, 3, 5 );

    cv::Mat_<double> X1;
    solve( A, B, X1, 1e-10, SparseMatrixCV::BICGSQ );

    cv::Mat_<double> X2;
    cv::solve( toCvMat(A), toCvMat(B), X2);

    test_equal( X1, X2 );
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    int flag = RUN_ALL_TESTS();
    system("pause");
    return flag;
}


