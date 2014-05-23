#include "SparseMatrixTest.h"

void SparseMatrixTest::SetUp()
{
    /* [19,  0, 21, 21,  0,  0;
    	12, 21,  0,  0,  0,  0;
    	 0, 12, 16,  0,  0,  0;
    	 0,  0,  0,  5, 21,  0;
    	12, 12,  0,  0, 18,  1]; */
    double non_zero_value_A1[13] = {19, 21, 21, 12, 21, 12, 16,  5, 21, 12, 12, 18, 1};
    int    col_index_A1[13] =     { 0,  2,  3,  0,  1,  1,  2,  3,  4,  0,  1,  4, 5};
    int    row_pointer_A1[6] =    { 0,          3,      5,      7,      9,        13};
    A1 = SparseMatrix(5, 6, non_zero_value_A1, col_index_A1, row_pointer_A1, 13);

    double non_zero_value_A2[13] ={19,-21, 21,-12, 21, 12, 16,  5, 21, 12, 12, 18, 1};
	int    col_index_A2[13] =     { 0,  2,  3,  0,  1,  1,  2,  2,  4,  0,  3,  4, 5};
	int    row_pointer_A2[6] =    { 0,          3,      5,      7,      9,        13};
	A2 = SparseMatrix(5, 6, non_zero_value_A2, col_index_A2, row_pointer_A2, 9);

    const int sizeB = 9;
    double non_zero_value_B[sizeB] = { 2,  1,  3,  2,  6,  8,  6,  8, 18};
    int    col_index_B[sizeB]      = { 0,  1,  2,  0,  1,  2,  0,  1,  2};
    int    row_pointer_B[4]    = { 0,          3,          6,          9};
    B = SparseMatrix(3, 3, non_zero_value_B, col_index_B, row_pointer_B, sizeB);
}

void SparseMatrixTest::TearDown()
{

}
