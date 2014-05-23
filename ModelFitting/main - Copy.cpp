#include <iostream>

using namespace std;

int main(void) {
    cout << "hello world" << endl;

    int sum = 0;
    for( int i=0; i<100; i++ ){
        sum += i;
    }
    cout << sum << endl;

    return 0;
}
