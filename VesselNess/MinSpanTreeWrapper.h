#pragma once

#include "MinSpanTree.h"

#include "Data3D.h"
#include "Vesselness.h"
#include "ImageProcessing.h"

namespace MinSpanTree {

	template<class EdgeType = Edge, class NodeType = char>
	class Graph3D : public Graph<EdgeType, NodeType> {
	public:
		int sx;
		int sy;
		int sz;
	
		Graph3D() : Graph( 0 ), sx( 0 ) , sy( 0 ) , sz( 0 ) {  }
		Graph3D( unsigned int x, unsigned int y, unsigned int z ) 
			: Graph( x*y*z ), sx( x ) , sy( y ) , sz( z ) {  }

		// get node id from position
		inline int nodeid(const Vec3i& pos ) {
			return nodeid(pos[0], pos[1], pos[2]); 
		}
		inline int nodeid(const int& x, const int& y, const int&z) {
			return z*sx*sy + y*sx + x;
		}
		inline Vec3i get_pos( const int& nodeid ) {
			return Vec3i( nodeid % (sx*sy) % sx, nodeid % (sx*sy) / sx, nodeid / (sx*sy) );
		}
	}; 

	void edge_tracing( Data3D<Vesselness_All>& src_vn, Graph3D<Edge>& tree, const float& thres1, const float& thres2  ){
		// construct the graph
		Graph3D<Edge> graph( src_vn.SX(), src_vn.SY(), src_vn.SZ() );

		// non-maximum suppression
		Data3D<Vesselness_Sig> res_nms; 
		IP::non_max_suppress( src_vn, res_nms );
		

		// 26 neighbourhood system
		Vec3i offset[26];
		for( int i=0; i<26; i++ ){
			int index = (i + 14) % 27;
			offset[i][0] = index/9%3 - 1;
			offset[i][1] = index/3%3 - 1;
			offset[i][2] = index/1%3 - 1;
		}

		// normailize the data based on vesselness response
		Data3D<float> src1d;
		res_nms.copyDimTo( src1d, 0 );
		IP::normalize( src1d, 1.0f );

		// find seed points with the 1st threshold
		int x, y, z;
		std::queue<Vec3i> myQueue1;
		Data3D<unsigned char> mask( res_nms.get_size() );
		for(z=0;z<res_nms.SZ();z++) for (y=0;y<res_nms.SY();y++) for(x=0;x<res_nms.SX();x++) {
			if( src1d.at(x,y,z) > thres1 ) {
				myQueue1.push( Vec3i(x,y,z) );
				mask.at(x,y,z) = 255;
			}
		} 
		
		// region growing base on the second threshold
		while( !myQueue1.empty() ){
			Vec3i pos = myQueue1.front(); myQueue1.pop();
			Vec3i off_pos;
			for( int i=0; i<26; i++ ) {
				off_pos = pos + offset[i];
				if( res_nms.isValid(off_pos) && !mask.at( off_pos ) && src1d.at(off_pos) > thres2 ){
					mask.at( off_pos ) = 255;
					myQueue1.push( off_pos );
				}
			}
		}

		// Now we have the mask of the center-line. We want to connect them as a minimum spinning tree. 
		// But before that, we need to know their conectivity based on breath first search
		DisjointSet djs( graph.num_nodes() );
		Data3D<unsigned char> isVisited( mask.get_size() );
		for(z=0;z<src_vn.SZ();z++) for (y=0;y<src_vn.SY();y++) for(x=0;x<src_vn.SX();x++) {
			// if this voxel belongs to the center-line and has not been labeled yet.
			// We will run a breadth-first search from this voxel to label all other labels that are connected to this one. 
			if( mask.at(x,y,z)==255 && !isVisited.at(x,y,z) ){ 
				// breadth-first search on this point
				std::queue<Vec3i> q;
				q.push( Vec3i(x,y,z) );
				while( !q.empty() ){
					Vec3i pos = q.front(); q.pop();
					for( int i=0; i<26; i++ ) { // 26 neighbourhood
						Vec3i off_pos = pos + offset[i];
						if( mask.isValid(off_pos) && mask.at( off_pos )==255 && !isVisited.at(off_pos) )
						{
							isVisited.at(off_pos) = 1;
							djs.merge( graph.nodeid(x,y,z), graph.nodeid(off_pos) );
							q.push( off_pos );
						} 
						if( mask.isValid(off_pos) && mask.at( off_pos )==255 ) {
							Edge e;
							e.node1 = graph.nodeid(pos);
							e.node2 = graph.nodeid(off_pos);
							if( e.node1 > e.node2 ) continue;
							e.weight = (float) Vec3i(x,y,z).dot(off_pos);
							graph.add_edge( e );
						}
					}
				} // endl of breath first search
			}
		}
		
		graph.get_min_span_tree( tree );
		tree.sx = graph.sx;
		tree.sy = graph.sy;
		tree.sz = graph.sz;
		return;





		const unsigned char ENDPOINT_YES = 255;
		const unsigned char ENDPOINT_NO  = 144;
		const unsigned char UN_DEFINED  = 0;   // undefined
		Data3D<unsigned char> endpoints_mask1( mask.get_size() );
		for(z=1;z<src_vn.SZ()-1;z++) for (y=1;y<src_vn.SY()-1;y++) for(x=1;x<src_vn.SX()-1;x++) {
			if( mask.at( x,y,z )==0 ) continue;
			if( endpoints_mask1.at(x,y,z)!=UN_DEFINED ) continue;
			// breath-first search begin
			std::queue<Vec3i> myQueue;
			myQueue.push( Vec3i(x,y,z) );
			while( !myQueue.empty() ) {
				Vec3i pos = myQueue.front(); myQueue.pop();
				// initial guess the this pos to be a endpoint
				endpoints_mask1.at( pos ) = ENDPOINT_YES;
				// transverse the neighbour hood system
				for( int i=0; i<26; i++ ) { 
					Vec3i off_pos = pos + offset[i];
					if( !mask.isValid( off_pos ) ) continue;
					if( mask.at(off_pos)==0 ) continue;
					if( endpoints_mask1.at(off_pos)==UN_DEFINED ) {
						myQueue.push( off_pos );
						endpoints_mask1.at( off_pos ) = ENDPOINT_YES; 
						endpoints_mask1.at( pos ) = ENDPOINT_NO;
					} else if( endpoints_mask1.at(off_pos)==ENDPOINT_YES ) { 
						endpoints_mask1.at( pos ) = ENDPOINT_NO;
					}
				}
			}
		}
		Data3D<unsigned char> endpoints_mask2( mask.get_size() );
		for(z=src_vn.SZ()-2;z>=1;z--) for(y=src_vn.SY()-2;y>=1;y--) for(x=src_vn.SX()-2; x>=1; x--) {
			if( mask.at( x,y,z )==0 ) continue;
			if( endpoints_mask2.at(x,y,z)!=UN_DEFINED ) continue;
			// breath-first search begin
			std::queue<Vec3i> myQueue;
			myQueue.push( Vec3i(x,y,z) );
			while( !myQueue.empty() ) {
				Vec3i pos = myQueue.front(); myQueue.pop();
				// initial guess the this pos to be a endpoint
				endpoints_mask2.at( pos ) = ENDPOINT_YES;
				// transverse the neighbour hood system
				for( int i=0; i<26; i++ ) { 
					Vec3i off_pos = pos + offset[i];
					if( !mask.isValid( off_pos ) ) continue;
					if( mask.at(off_pos)==0 ) continue;
					if( endpoints_mask2.at(off_pos)==UN_DEFINED ) {
						myQueue.push( off_pos );
						endpoints_mask2.at( off_pos ) = ENDPOINT_YES; 
						endpoints_mask2.at( pos ) = ENDPOINT_NO;
					} else if( endpoints_mask2.at(off_pos)==ENDPOINT_YES ) { 
						endpoints_mask2.at( pos ) = ENDPOINT_NO;
					}
				}
			}
		}

		vector<Vec3i> endpoints; // endpoints are the points that have only one neighbour
		for(z=1;z<mask.SZ()-1;z++) for (y=1;y<mask.SY()-1;y++) for(x=1;x<mask.SX()-1;x++) {
			if( endpoints_mask1.at(x,y,z)==ENDPOINT_YES || endpoints_mask2.at(x,y,z)==ENDPOINT_YES ) {
				if( src1d.at(x,y,z)> (0.5f*thres1+0.5f*thres2) ) {
					endpoints.push_back( Vec3i(x,y,z) );
				}
			}
		} 

		// two small data structures for the use priority_queue
		class Dis_Pos {
		private: 
			float dist;
			Vec3i to_pos;
		public:
			Dis_Pos( const float& distance, const Vec3i& position ) 
				: dist(distance) 
				, to_pos(position) { }
			inline bool operator<( const Dis_Pos& right ) const { 
				// for the use of priority_queue, we reverse the sign of comparison from '<' to '>'
				return ( this->getDist() > right.getDist() ); 
			} 
			inline const float& getDist(void) const { return dist; }
			inline const Vec3i& getToPos(void) const { return to_pos; }
		};
		class Dis_Pos_Pos : public Dis_Pos {
		private:
			Vec3i from_pos;
		public:
			Dis_Pos_Pos( const Dis_Pos& dis_pos, const Vec3i& from_posistion ) 
				: Dis_Pos(dis_pos), from_pos(from_posistion) { }
			inline const Vec3i& getFromPos(void) const { return from_pos; }
		};


		std::priority_queue< Dis_Pos_Pos > min_dis_queue; 
		const unsigned char VISITED_YES = 255;
		const unsigned char VISITED_N0  = 0;
		vector<Vec3i>::iterator it;
		for( it=endpoints.begin(); it<endpoints.end(); it++ ) {
			const Vec3i& from_pos = *it;
			std::priority_queue< Dis_Pos > myQueue; 
			myQueue.push( Dis_Pos( 0.0f, from_pos) );

			// breath first search
			isVisited.reset(); // set all the data to 0. 
			isVisited.at( from_pos ) = 255;
			bool to_pos_fount = false; 
			while( !myQueue.empty() && !to_pos_fount ) {
				Dis_Pos dis_pos = myQueue.top(); myQueue.pop(); 
				for( int i=0; i<26; i++ ) { 
					// get the propogate position
					Vec3i to_pos = offset[i] + dis_pos.getToPos();
					if( !isVisited.isValid(to_pos) ) continue;
					if(  isVisited.at(to_pos)==VISITED_YES ) continue;

					if( djs.find( graph.nodeid(to_pos) )!=djs.find( graph.nodeid(from_pos) ) ) {
						Vec3i dif = to_pos - from_pos;
						float dist = sqrt( 1.0f*dif[0]*dif[0] + dif[1]*dif[1] + dif[2]*dif[2] );
						if( dist > 7.0f ) continue; // prevent from searching too far aways

						if( mask.at( to_pos )==0 || src1d.at(to_pos)< thres1 ) {	
							// if this voxel belongs to a background set
							myQueue.push( Dis_Pos(dist, to_pos) );
							isVisited.at( to_pos ) = VISITED_YES;
						}

						else {
							// if this voxels have different setid
							to_pos_fount = true;
							float ratio = src1d.at(from_pos) / src1d.at(to_pos);
							if( ratio<1.0f ) ratio = 1 / ratio;
							dist += ratio;
							min_dis_queue.push( Dis_Pos_Pos(Dis_Pos(dist*ratio, to_pos), from_pos) );
							break;
						}
					}
				}
			}
		}

		while( !min_dis_queue.empty() ){
			Dis_Pos_Pos dpp = min_dis_queue.top(); min_dis_queue.pop();
			const Vec3i& from_pos = dpp.getFromPos();
			const Vec3i& to_pos = dpp.getToPos();
			const float& dist = dpp.getDist(); 
			if( djs.find(graph.nodeid(to_pos))==djs.find(graph.nodeid(from_pos)) ) continue;

			djs.merge( graph.nodeid(to_pos), graph.nodeid(from_pos) ); 
			Edge e;
			e.node1 = graph.nodeid(to_pos);
			e.node2 = graph.nodeid(from_pos);
			e.weight = 1.0;//  I will take care of this later
			graph.add_edge( e );
		}

		tree = graph;
	}
}