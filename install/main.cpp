#include <iostream>
#include <fstream>
#include <vector>

using namespace std;



// The following color is defined using [ANSI colour codes](http://en.wikipedia.org/wiki/ANSI_escape_code)
const string RED = "\033[1;31m";
const string GREEN = "\033[1;32m";
const string BLACK = "\x1b[0;49m";

int main()
{

    vector<std::pair<string, string> > files2copy
    {
        std::pair<string, string>("../SparseMatrixCV/bin/Release/libSparseMatrixCV.a",
                                  "../libs/libSparseMatrixCV.a")
    };

    std::ifstream  src;
    std::ofstream  dst;
    for( unsigned i=0; i<files2copy.size(); i++ )
    {
        const string& from = files2copy[i].first;
        const string& to = files2copy[i].second;

        src.open( from );
        dst.open( to );
        if( src.fail() ) {
            cout << RED << " [FAIL   ] " << BLACK;
            cout << "Unable to open file '" << from << "'" << endl;
        } else {
            dst << src.rdbuf();
            cout << GREEN << " [SUCCESS] " << BLACK;
            cout << "File copied from '" << from  << "'";
            cout << " to '" << to << "'" << endl;
        }
    }
}
