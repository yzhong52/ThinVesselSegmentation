SparseMatrixCV
========================

OpenCV warper for SparseMatrix. For matrix multiplication between sparse matrix and dense matrix. For example, 

	template <class _Tp, int m, int n>
	SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm ); 

	Mat_<double> operator*( const SparseMatrixCV& sm, const Mat_<double>& sm ); 

More functions maybe added in the future. 