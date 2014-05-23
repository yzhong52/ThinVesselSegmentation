#include "gtest/gtest.h"

#include "SparseMatrixTest.h"

#include <iostream>
using namespace std;


TEST(SparseMatrixTest, Constructor){
        /* [19,  0, 21, 21,  0,  0;
    	12, 21,  0,  0,  0,  0;
    	 0, 12, 16,  0,  0,  0;
    	 0,  0,  0,  5, 21,  0;
    	12, 12,  0,  0, 18,  1]; */
    const int N = 13;
    double non_zero_value_A1[N] = {19, 21, 21, 12, 21, 12, 16,  5, 21, 12, 12, 18, 1};
    int    col_index_A1[N] =     { 0,  2,  3,  0,  1,  1,  2,  3,  4,  0,  1,  4, 5};
    int    row_pointer_A1[6] =    { 0,          3,      5,      7,      9,        13};
    SparseMatrix A1(5, 6, non_zero_value_A1, col_index_A1, row_pointer_A1, N);

    A1.print(cout);

    cout << A1 << endl;
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    int flag = RUN_ALL_TESTS();
    system("pause");
    return flag;
}


