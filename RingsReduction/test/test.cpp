#include "gtest/gtest.h"

#include "RingsReductionTest.h"
#include "../RingsReduction.h"

#include <iostream>
using namespace std;


TEST( Test, dist )
{
    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 1, 1, 5, 5), 0.0 );
    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 1, 1, 4, 5), 0.0 );
    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 1, 1, 5, 2), 0.0 );
    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 1, 1, -2, 3), 0.0 );
    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 1, 1, -2, -4), 0.0 );

    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 0, 0, 0, 1 ), 1 );
    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 0, 0, 1, 0 ), 1 );
    EXPECT_DOUBLE_EQ( RR::dist( 1, 1, 0, 1, 1, 1 ), sqrt(2.0)/2 );
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    int flag = RUN_ALL_TESTS();
    return flag;
}


