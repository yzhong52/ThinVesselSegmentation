#ifndef SPARSEMATRIXCVTEST_H
#define SPARSEMATRIXCVTEST_H

#include "gtest/gtest.h"
#include "../SparseMatrix.h"

class SparseMatrixTest : public testing::Test
{
protected:
    SparseMatrix A1, A2, B, C1, C2;
    virtual void SetUp();
    virtual void TearDown();
};


#endif // SPARSEMATRIXCVTEST_H
