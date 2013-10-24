#include "Neigbour3D.h"

Vec3i Neigbour3D::dir[26];
Vec3i Neigbour3D::normal[26][9];

Neigbour3D::Neigbour3D(void)
{
	// If we put the directions in this order
	for( int i=0; i<26; i++ ){
		int index = (i + 14) % 27;
		dir[i][0] = index/9%3 - 1;
		dir[i][1] = index/3%3 - 1;
		dir[i][2] = index/1%3 - 1;
	}

	for( int i=0; i<26; i++ ){
		cout << i << "=\t" << dir[i] << "\t=" << index(dir[i]) << endl;
	}

	memset( normal, 0, sizeof(int)*26*9 );
	// Setting offsets that are perpendicular to dirs
	for( int i=0; i<26; i++ ) {
		int ni = 0;
		// there are at most 8 such offsets for each direciton
		for( int j=0; j<26; j++ ) {
			// multiply the two directions
			int temp = 
				dir[i][0] * dir[j][0] +
				dir[i][1] * dir[j][1] +
				dir[i][2] * dir[j][2];
			// the temp is 0, store this direciton
			if( temp==0 ) {
				normal[i][ni++] = dir[j];
			}
		}
	}

	for( int i=0; i<26; i++ ){
		for( int j=0; j<9; j++ ) cout << normal[i][j] << "\t"; 
		cout << endl << endl;
	} 
}


int Neigbour3D::index( const Vec3i& d ){
	smart_assert( d[0]<=1 && d[0]>=-1, "Invalid Dir" );
	smart_assert( d[1]<=1 && d[1]>=-1, "Invalid Dir" );
	smart_assert( d[2]<=1 && d[2]>=-1, "Invalid Dir" );
	smart_assert( d[0] || d[1] || d[2], "Invalid Dir" );

	int i = (d[2]+1) + (d[1]+1)*3 + (d[0]+1)*9;
	i = ( i-14+27 ) % 27;
	return i;
}

Neigbour3D::~Neigbour3D(void)
{
}
