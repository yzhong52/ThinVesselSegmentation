:: set deployment path
set targetdir=".\libs"

for %%a in (
		".\SparseMatrix\bin\Debug\*.a"
		".\SparseMatrixCV\bin\Debug\*.a"
	) do (
    	copy %%a %targetdir%
	)

PAUSE