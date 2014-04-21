#include "SparseMatrixCV.h"

#ifdef _DEBUG
#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif

#include <vector>

using namespace std;

SparseMatrixCV::SparseMatrixCV( int nrow, int ncol, const int index[][2], const double value[], int N )
{
	struct Three{
		Three( int r, int c, double v) : row(r), col(c), val(v) { }
		int row, col;
		double val;
		inline bool operator<( const Three& t ) { return row < t.row || row==t.row && col < t.col; }
	};
	vector<Three> initdata; 
	for( int i=0; i<N; i++ ) {
		initdata.push_back( Three( index[i][0], index[i][1], value[i] ) ); 
	}

	std::sort( initdata.begin(), initdata.end() ); 


	double* non_zero_value  = new double[N]; 
	int* col_index  = new int[N]; 
	int* row_pointer = new int[nrow+1]; 

	int previous_row = -1; 
	for( int i=0; i<N; i++ ){
		non_zero_value[i] = initdata[i].val;
		col_index[i] = initdata[i].col;
		if( initdata[i].row != previous_row ) {
			row_pointer[++previous_row] = i; 
		}
	}

	// re-constuct the matrix with give data
	delete this->data; 
	this->data = new SparseMatrixData( nrow, ncol, 
		(const double*) non_zero_value, 
		(const int* )   col_index, 
		(const int* )   row_pointer, N ); 

	delete[] row_pointer;
	delete[] col_index;
	delete[] non_zero_value;
}