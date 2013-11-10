#pragma once
#include <iostream>
#include <vector>

// Yuchen: This Graph is contructed using Adjacency List

template<class Edge_Type>
struct Adjancy {
	int node;
	Edge_Type edge;
	Adjancy( int node, Edge_Type edge )
		: node( node ), edge( edge ) { } 
	Adjancy() { }
}; 

template<class Edge_Type>
class AjList_Graph
{
	// private variables
	int _num_node;
	std::vector< Adjancy<Edge_Type> >* adjacency_list;

public:
	// Constructor and Destroctor
	AjList_Graph() : _num_node(0), adjacency_list(NULL) { }
	AjList_Graph( int num_node );
	~AjList_Graph( void ); 

	void resize( int num_node );

	// add an edge to a graph
	void add_edge(int node1, int node2, Edge_Type edge );


	// Getters
	int get_num_node() const { return _num_node; }

	template<class T>
	friend std::ostream& operator<<( std::ostream& out, const AjList_Graph<T>& g );
};

template<class Edge_Type>
AjList_Graph<Edge_Type>::AjList_Graph(int num_node)
	: _num_node( num_node )
{
	adjacency_list = new std::vector< Adjancy<Edge_Type> >[ _num_node ];
}


template<class Edge_Type>
AjList_Graph<Edge_Type>::~AjList_Graph(void)
{
	delete[] adjacency_list;
}

template<class Edge_Type>
void AjList_Graph<Edge_Type>::resize( int num_node ){
	_num_node = num_node;
	delete[] adjacency_list;
	adjacency_list = new std::vector< Adjancy<Edge_Type> >[ _num_node ];
}

template<class Edge_Type>
void AjList_Graph<Edge_Type>::add_edge(int node1, int node2, Edge_Type edge ){
	if( node1<0 || node1>=_num_node ||  node2<0 || node2>=_num_node ){
		std::cerr << "Add Edge Failed: node id is invalid." << std::endl;
	}
	adjacency_list[node1].push_back( Adjancy<Edge_Type>(node2, edge) );
	adjacency_list[node2].push_back( Adjancy<Edge_Type>(node1, edge) );
}


template<class T>
std::ostream& operator<<( std::ostream& out, const AjList_Graph<T>& g ){
	
	for( int i=0; i<g._num_node; i++ ) {
		std::cout<< "Node " << i << ": " << endl;
		vector< Adjancy<T> > adj_list = g.adjacency_list[i];
		vector< Adjancy<T> >::iterator it;
		for( it = adj_list.begin(); it < adj_list.end(); it++ ){
			out << " -" << (it->node) << " " << (it->edge);
			out << endl;
		}
	}
	return out;
}
