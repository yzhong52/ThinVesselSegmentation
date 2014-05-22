:: set deployment path
set targetdir=".\libs"

for %%a in (
		".\SparseMatrix\bin\Debug\*.*"
		".\SparseMatrixCV\bin\Debug\*.*"
	) do (
    	copy %%a %targetdir%
	)

PAUSE