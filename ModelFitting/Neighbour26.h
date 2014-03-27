#pragma once
class Neighbour26
{
	int offset[26][3]; 
	Neighbour26(void)
	{
		for( int i=0; i<26; i++ ){
			int index = (i + 14) % 27;
			offset[i][0] = index/9%3 - 1;
			offset[i][1] = index/3%3 - 1;
			offset[i][2] = index/1%3 - 1;
		}
	}

	virtual ~Neighbour26(void)
	{
	}

	static Neighbour26& getInstance() {
		static Neighbour26 instance; 
		return instance; 
	}
public:

	static void at( int index, int& x, int& y, int& z ) 
	{
		if( index<0 || index>=26 ) {
			cout << "Index Error for 26 neigbours" << endl; 
			system("pause"); 
			x = y = z = 0; 
			return; 
		}
		x = getInstance().offset[index][0]; 
		y = getInstance().offset[index][1]; 
		z = getInstance().offset[index][2]; 
	}

	static void getNeigbour( int index, 
		const int& old_x, 
		const int& old_y, 
		const int& old_z, 
		int& x, int& y, int& z ) 
	{
		if( index<0 || index>=26 ) {
			cout << "Index Error for 26 neigbours" << endl; 
			system("pause"); 
			x = old_x;
			y = old_y; 
			z = old_z; 
			return; 
		}
		x = getInstance().offset[index][0] + old_x; 
		y = getInstance().offset[index][1] + old_y; 
		z = getInstance().offset[index][2] + old_z; 
	}
};

