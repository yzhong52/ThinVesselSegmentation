#include "gtest/gtest.h"

#include "SparseMatrixTest.h"

#include <iostream>
using namespace std;


TEST_F(SparseMatrixTest, Addition)
{
    const int row = 5;
    const int col = 6;
    double expetec[row][col] =
    {
        {38,    0,    0,   42,    0,    0},
        { 0,   42,    0,    0,    0,    0},
        { 0,   24,   32,    0,    0,    0},
        { 0,    0,    5,    5,   42,    0},
        {12,   12,    0,    0,   18,    1}
    };

    cout << "A1 = " << A1 << endl;
    cout << "A2 = " << A2 << endl;
    cout << "A1 + A2 = " << A1 + A2 << endl;

    cout << "Expected result: " << endl;
    print( expetec );

    test_equal( expetec, A1 + A2 );
}


TEST_F(SparseMatrixTest, Subtraction)
{
    const int row = 5;
    const int col = 6;
    double expetec[row][col] =
    {
        { 0,    0,   42,    0,    0,    0},
        {24,    0,    0,    0,    0,    0},
        { 0,    0,    0,    0,    0,    0},
        { 0,    0,   -5,    5,    0,    0},
        {12,   12,    0,    0,   18,    1}
    };

    cout << "A1 = " << A1 << endl;
    cout << "A2 = " << A2 << endl;
    cout << "A1 - A2 = " << A1 - A2 << endl;

    cout << "Expected result: " << endl;
    print( expetec );

    test_equal( expetec, A1 - A2 );
}


TEST_F(SparseMatrixTest, Multiplication)
{
//    const int row = 5;
//    const int col = 5;
//    double expetec[row][col] = {
//        { 0,    266,   0,  105,  441 },
//        { 252,  168,   0,    0,  147 },
//        { 144,    0,   0,    0,   84 },
//        { 252,  252,   0,   25,  483 },
//        { 360,  384,   0,    0,  409 } };
//
//    cout << A1 * A2 << endl;
//    test_equal( expetec, A1 * A2 );
}


TEST_F(SparseMatrixTest, MultiplyTranspose)
{
    const int row = 5;
    const int col = 5;
    double expected[row][col] =
    {
        {1243,        228,        336,        105,        228},
        { 228 ,       585 ,       252 ,         0 ,       396},
        { 336  ,      252  ,      400  ,        0  ,      144},
        { 105   ,       0   ,       0   ,     466   ,     378},
        { 228    ,    396    ,    144    ,    378    ,    613}
    };

    cout << "A1: " << A1 << endl << endl;

    cout << "A1 * A1.t(): " << A1 * A1.t() << endl << endl;

    cout << "Expected result: " << endl;
    print( expected );

    test_equal( expected, A1 * A1.t() );
}



TEST_F(SparseMatrixTest, MemoryLeak)
{
    cout << "Hey! We are now testing memory leak. ";
    cout << "We need your help. ";
    cout << "Please open your system monitor and observe the memory usage. ";
    cout << "The memory should not be increasing. ";
    cout << "If it does, then it means there are memory leaks. ";
    cout << "Please kill the testing program as soon as you can. " << endl;

    cout << "Ready? Please enter to continue... " << endl;
    cin.get();

    const int num_inter = 10e7;
    for( int i=0; i<num_inter; i++ ){
        if( i%(num_inter/1000)==0 ) cout << "\r" << 100.0 * i / num_inter << "%"; cout.flush();
        SparseMatrix res = A1 * A1.t();
        SparseMatrix res1 = A1.t() * A1;
    }
    cout << endl;
}




int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    int flag = RUN_ALL_TESTS();
    return flag;
}


