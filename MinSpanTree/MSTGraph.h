#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include "MSTEdge.h"
#include "DisjointSet.h"

namespace MST
{

template<class EdgeType = MST::Edge, class NodeType = char>
class Graph
{
private:
    // number of nodes
    std::vector<NodeType> nodes;
    // Edges of the graph
    std::priority_queue<EdgeType> edges;

public:
    // Constructor & Destructor
    Graph( ) { }

    Graph( unsigned int num_node )
    {
        nodes.resize( num_node );
    }

    // reset the graph and clear all the edges
    inline void reset( int num_node )
    {
        // resize the nodes
        nodes.resize( num_node );
        // clear the edges
        clear_edges();
    }

    inline void reset( const std::vector<NodeType>& new_nodes )
    {
        // update the nodes
        nodes = new_nodes;
        // clear the edges
        clear_edges();
    }

    inline void reset( const std::vector<NodeType>& new_nodes,
                       const std::priority_queue<EdgeType>& new_edges )
    {
        // update the nodes
        nodes = new_nodes;
        // update the edges
        edges = new_edges;
    }



    inline void clear_edges(void)
    {
        // clear the edges
        edges = std::priority_queue<EdgeType>();
    }

    // add an edge to a graph
    inline void add_edge( EdgeType edge )
    {
        edges.push( edge );
    }

    // add an node to a graph
    inline void add_node( NodeType node )
    {
        nodes.push_back( node );
    }

    //getters
    inline const std::priority_queue<EdgeType>& get_edges( void ) const
    {
        return edges;
    }

    inline const std::vector<NodeType>& get_nodes( void ) const
    {
        return nodes;
    }

    inline const NodeType& get_node(const int& i) const
    {
        return nodes[i];
    }

    inline const EdgeType& get_edge(const int& i) const
    {
        return *(&edges.top()+i);
    }

    unsigned num_edges(void) const
    {
        return (unsigned)edges.size();
    }

    unsigned num_nodes(void) const
    {
        return (unsigned)nodes.size();
    }

    // get a minimum spanning tree of the current graph
    void get_min_span_tree( Graph<EdgeType, NodeType>& dst ) const;

    // print graph for debug
    template<class E, class N>
    friend std::ostream& operator<<( std::ostream& out, Graph<E, N>& g );
};


template<class EdgeType, class NodeType>
void Graph<EdgeType, NodeType>::get_min_span_tree( Graph<EdgeType, NodeType>& dst ) const
{
    dst.reset( this->get_nodes() );
    DisjointSet djs( this->num_nodes() );

    // build edges
    std::priority_queue<EdgeType> edges = this->get_edges();
    while( !edges.empty() && dst.num_edges()<dst.num_nodes()-1 )
    {
        EdgeType e = edges.top();
        const int sid1 = djs.find( e.node1 );
        const int sid2 = djs.find( e.node2 );
        if( sid1 != sid2 )
        {
            dst.add_edge( e );
            djs.merge( sid1, sid2 );
        }
        edges.pop();
    }
}

template<class E, class N>
std::ostream& operator<<( std::ostream& out, Graph<E, N>& g )
{
    if( g.edges.empty() )
    {
        out << "OMG: There is currently no edges in graph! ";
        return out;
    }

    out << "Number of Edge: " << g.num_edges() << std::endl;

    // Transverse the edges and print them
    const E* pt_edge = &g.edges.top();
    for( unsigned int i=0; i<g.edges.size(); i++ )
    {
        std::cout << (*pt_edge) << std::endl;
        pt_edge++;
    }
    return out;
}

} // end of namespace
