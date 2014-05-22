:: set deployment path
set targetdir=".\libs"

for %%a in (
		".\SparseMatrix\bin\Release\*.*"
	) do (
    	copy %%a %targetdir%
	)

PAUSE