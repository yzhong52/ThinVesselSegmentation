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
public:
	static void at( int index, int& x, int& y, int& z ) {
		static Neighbour26 instance; 
		if( index>=0 && index<26 ) {
			x = instance.offset[index][0]; 
			y = instance.offset[index][1]; 
			z = instance.offset[index][2]; 
		} else {
			x = y = z = 0; 
		}
	}
};

