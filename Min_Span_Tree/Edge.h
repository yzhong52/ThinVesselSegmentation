#pragma once
#include <iostream>

class Edge
{
public:
	Edge(void);
	~Edge(void);

	int node1, node2;
	float weight;

	// operator overflow
	bool operator<( Edge right ) const {
		return (this->weight > right.weight);
	}

	Edge( int node1, int node2, float weight ) 
		: node1( node1 ), node2( node2 ), weight( weight ) { }

	friend std::ostream& operator<<( std::ostream& out, const Edge& e );
	friend void print(const Edge& e);
};


class Vec3f{
public:
	float x, y, z;
	// constructor
	Vec3f(const Vec3f&v = Vec3f(0,0,0) ) : x(v.x), y(v.y), z(v.z) { }
	Vec3f( float x, float y, float z ) : x(x), y(y), z(z) { }
	// operator
	float dot( const Vec3f& v ) {
		return x * v.x + y * v.y + z * v.z;
	} 
	Vec3f operator-( const Vec3f& v) {
		return Vec3f( x-v.x, y-v.y, z-v.z);
	}
	Vec3f operator+( const Vec3f& v) {
		return Vec3f( x+v.x, y+v.y, z+v.z);
	}
	Vec3f operator*( const float& v) {
		return Vec3f( v*x, v*y, v*z );
	}
};



struct Line{
	Vec3f p1, p2;
	Line( ) : p1( Vec3f(0,0,0) ), p2( Vec3f(0,0,0) ) { }
	Line(const Line& l ) : p1(l.p1), p2(l.p2) { }
	~Line(){}
};



class Edge_Ext : public Edge {
public:
	Vec3f p1, p2;
};

