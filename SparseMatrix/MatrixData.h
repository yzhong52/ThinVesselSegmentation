#ifndef MATRIXDATA_H
#define MATRIXDATA_H

class MatrixData
{
    friend class SparseMatrixData;
private:
    int nnz;       // number of non-zero value
    double *nzval; // pointer to array of nonzero values
    union
    {
        // row order representation
        struct
        {
            int *colind; // pointer to array of columns indices of the nonzeros
            int *rowptr; // pointer to array of beginning of rows in nzval[] and colind[]
        };
        // column order representtaion
        struct
        {
            int *rowind; // pointer to array of row indices of the nonzeros
            int *colptr; // pointer to array of beginning of columns in nzval[], and rowind[]
        };
    };

    MatrixData()
        : nnz(0), nzval(nullptr), colind(nullptr), rowptr(nullptr) {}

    inline void release()
    {
        delete[] nzval;
        delete[] colind;
        delete[] rowptr;
        clear();
    }

    inline void clear()
    {
        nnz = 0;
        nzval = nullptr;
        colind = nullptr;
        rowptr = nullptr;
    }

public:
    bool isEmpty() const { return nnz==0; }
};


#endif // MATRIXDATA_H
