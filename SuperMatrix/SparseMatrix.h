#pragma once

// http://crd-legacy.lbl.gov/~xiaoye/SuperLU/

#include "SuperLU_4.3\SRC\slu_ddefs.h"

class SparseMatrix
{
	SuperMatrix* col;
	SuperMatrix* row;
public:
	SparseMatrix(void);
	~SparseMatrix(void);

	friend void solve( const SuperMatrix& A, const double* B, double* X ); 
};
