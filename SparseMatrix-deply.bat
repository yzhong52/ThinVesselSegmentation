:: set deployment path
set targetdir=".\libs"

for %%a in (
		".\SparseMatrix\bin\Release\*.a"
		".\SparseMatrixCV\bin\Release\*.a"
	) do (
    	copy %%a %targetdir%
	)

PAUSE