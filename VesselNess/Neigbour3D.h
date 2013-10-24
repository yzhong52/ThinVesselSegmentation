#pragma once
#include "stdafx.h"
class Neigbour3D
{
public:
	static Neigbour3D& getInstance() {
		static Neigbour3D instance;
		return instance;
	}

	static Vec3i dir[26];
	static Vec3i normal[26][9];
	static int index( const Vec3i& d );
private:
	Neigbour3D(void);
	virtual ~Neigbour3D(void);

	// Dont forget to declare these two. You want to make sure they
	// are unaccessable otherwise you may accidently get copies of
	// your singleton appearing.
	Neigbour3D( Neigbour3D const& );     // Don't Implement
	void operator=( Neigbour3D const& ); // Don't implement
};

