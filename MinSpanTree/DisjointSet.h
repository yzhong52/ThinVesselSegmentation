#ifndef MST_DISJOINT_SET_H
#define MST_DISJOINT_SET_H

#include <iostream>
/* A disjoint-set data structure is a data structure that keeps track of a set
   of elements partitioned into a number of disjoint (non-overlapping) subsets.
   A union-find algorithm is an algorithm that performs two useful operations
   on such a data structure.*/

class DisjointSet
{
public:
    /// Constructor
    DisjointSet();
    DisjointSet( int n_size );
    DisjointSet( const DisjointSet& djs );
    DisjointSet& operator=( const DisjointSet& djs );

    /// Destructor
    ~DisjointSet();

    /// Get the labeling at index i
    inline int operator[]( const int& i ) const;

    /// Find: Determine which subset a particular element is in. This can be used
    /// to determinine if two elements are in the same subset.
    inline int find(int id) const;

    /// Union: Join two subsets into a single subset
    inline void merge( int id1, int id2 );

    /// For Debug
    friend std::ostream& operator<<( std::ostream& out, const DisjointSet& djs );

    inline const int& get_size( void ) const
    {
        return size;
    }

private:
    /// Size of the set
    int size;
    /// Labeling of each element
    mutable int *data;
};


inline int DisjointSet::operator[]( const int& i ) const
{
    return data[i];
}

inline int DisjointSet::find(int id) const
{
    if( data[id] == -1 )
    {
        return id;
    }
    else
    {
        data[id] = find( data[id] );
        return data[id];
    }
}

inline void DisjointSet::merge( int id1, int id2 )
{
    if( data[id1]!=-1 ) id1 = find(id1);
    if( data[id2]!=-1 ) id2 = find(id2);
    if( id1==id2 ) return;
    data[id1] = id2;
}

#endif // MST_DISJOINT_SET_H


