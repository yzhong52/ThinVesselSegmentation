#ifndef SPARSEMATRIXCVTEST_H
#define SPARSEMATRIXCVTEST_H

#include "gtest/gtest.h"
#include "../RingsReduction.h"

#include <iostream>

class SparseMatrixTest : public testing::Test
{
protected:
    virtual void SetUp();
    virtual void TearDown();

};

#endif // SPARSEMATRIXCVTEST_H
