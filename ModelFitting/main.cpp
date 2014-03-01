// ModelFitting.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <Windows.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "Line3D.h"

// This project is build after VesselNess. 
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess. 
#include "Data3D.h" 
#include "GLViwerModel.h"
#include "MinSpanTree.h" 

// For the use of graph cut
#include "GCoptimization.h"
typedef GCoptimization GC; 

GLViwerModel ver;



// TODO: Fix this function.
int main(int argc, char* argv[])
{
	//vector<Vec3i> points; 
	//points.push_back( Vec3i(0,0,0) );
	//points.push_back( Vec3i(1,2,3) );

	//Vec6f line; 
	//cv::fitLine( points, line, CV_DIST_L2, 0, 0.01, 0.01 ); 
	//Vec3f line1( line[0], line[1], line[2] );
	//cout << line << endl; 
	//cout << line1 << endl; 
	//float len = sqrt( line1.dot( line1 ) );
	//cout << len << endl; 
	//cout << line1/len << endl; 

	//Vec3f line2( 1, 2, 3 );
	//float len2 = sqrt( line2.dot( line2 ) );
	//cout << len2 << endl;
	//cout << line2/len2 << endl; 

	//return 0; 

	CreateDirectory(L"./output", NULL);
	
	Data3D<short> im_short;
	im_short.load( "../data/data15.data" );
	//im_short.reset();
	//for(int i=0; i<im_short.SX(); i++ ) im_short.at(i, 50, 50) = 5000;

	Data3D<short> im_char( im_short.get_size() );

	//////////////////////////////////////////////////
	// Line Fitting
	//////////////////////////////////////////////////
	// Initial Sampling
	const int num_init_labels = 10; 
	vector<Line3D> lines; // three float of geometric locations and three for directions
	for( int i=0; i<num_init_labels; i++ ){
		Line3D line;
		// position of the line
		line.setPos( Vec3f(
			float( rand() % im_short.SX() ), 
			float( rand() % im_short.SY() ),
			float( rand() % im_short.SZ() ) )); 
		// direction of the line 
		line.setDir( Vec3f(
			(float) rand()+1, 
			(float) rand()+1, 
			(float) rand()+1 )); 
		lines.push_back( line );
	}
	

	try{
		GC::EnergyType energy_before = -1;

		for( int i=0; i<150; i++ ) { 
			GCoptimizationGeneralGraph gc( im_short.SX()*im_short.SY()*im_short.SZ(), (int)lines.size()+1 );

			////////////////////////////////////////////////
			// Data Costs
			////////////////////////////////////////////////
			for( int z=0; z<im_short.SZ(); z++ ) {
				for( int y=0; y<im_short.SY(); y++ ) {
					for( int x=0; x<im_short.SX(); x++ ) {		
						GC::SiteID site = x + y * im_short.SX() + z * im_short.SY() * im_short.SX();
						
						// Data Cost (Part I) - Color Conssitancy
						// TODO make this a parameter to tune
						GC::EnergyTermType datacost = 3500-im_short.at(x,y,z); 

						// data cost to general label
						GC::SiteID label;
						for( label = 0; label < lines.size(); label++ ){
							const Line3D& line = lines[label];
							// Data Cost (Part II) - Position Conssitancy
							// distance from a point to a line
							float dist = line.distanceToLine( Vec3f(1.0f*x,1.0f*y,1.0f*z) );
							// log likelihood based on the distance
							GC::EnergyTermType loglikelihood = dist*dist/( 2*line.sigma*line.sigma );
							loglikelihood += log( line.sigma ); 
							static double C = 0.5 * log( 2*M_PI );
							loglikelihood += C; 
							// TODO make 100 a parameter to tune
							gc.setDataCost( site, label, datacost + loglikelihood * 100 );
						}

						// data cost to background/outlier label
						gc.setDataCost( site, label, -datacost );
					}
				}
			}

			////////////////////////////////////////////////
			// Smooth Cost
			////////////////////////////////////////////////
			// ... 

			cout << "Iteration: " << i << ". Fitting Begin. Please Wait..."; 
			gc.expansion(1); // run expansion for 1 iterations. For swap use gc->swap(num_iterations);
			GC::EnergyType cur_energy = gc.compute_energy();
			if ( energy_before==cur_energy ) { 
				cout << endl << "Energy is not changing. " << endl; break; 
			} else {
				energy_before = cur_energy; 
			}
			cout << "Done. " << endl;
			
			int count1 = 0, count2 = 0; 
			for( int x=0; x<im_short.SX(); x++ ) {
				for( int y=0; y<im_short.SY(); y++ ) {
					for( int z=0; z<im_short.SZ(); z++ ) {						
						GC::SiteID site = x + y * im_short.SX() + z * im_short.SY() * im_short.SX();
						im_char.at(x,y,z) = 1-gc.whatLabel( site );
						if( im_char.at(x,y,z)==0 ) count1++;
						else				       count2++;
					}
				}
			}
			cout << "count1 = " << count1 << endl;
			cout << "count2 = " << count2 << endl;

			////////////////////////////////////////////////
			// Re-estimation
			////////////////////////////////////////////////
			// gether the points
			vector<vector<Vec3i> > points_set = vector<vector<Vec3i> >( lines.size() ); 
			for( int x=0; x<im_short.SX(); x++ ) {
				for( int y=0; y<im_short.SY(); y++ ) {
					for( int z=0; z<im_short.SZ(); z++ ) {
						GC::SiteID site = x + y * im_short.SX() + z * im_short.SY() * im_short.SX();
						GC::SiteID label = gc.whatLabel( site );
						if( label<lines.size() ){
							points_set[ label ].push_back( Vec3i(x,y,z) );
						}
					}
				}
			}
			// Line Fitting With Least Square 
			vector<vector<Vec3i> >::iterator psit = points_set.begin();
			vector<Line3D>::iterator lit  = lines.begin();
			while( psit<points_set.end() ) { 
				if( psit->size() ) { 
					Vec6f line; 
					cv::fitLine( *psit, line, CV_DIST_L2, 0, 0.01, 0.01);
					// update the 
					lit->setPos( Vec3f(&line[0]) );
					lit->setDir( Vec3f(&line[3]) );
					// we also need to update sigma though
					lit->sigma = 1.0f; 
				} 
				psit++; lit++; 
			}

			ver.addModel( &gc, lines, im_short.get_size() );

			// Remove unused models 
			vector<Line3D> newLines; 
			for( int i=0; i<points_set.size(); i++ ) { 
				if( points_set[i].size() ) { 
					newLines.push_back( lines[i] ); 
				}
			}
			newLines.push_back( lines.back() ); 
			lines = newLines; 
			break; 
		}
	}
	catch (GCException e){
		e.Report();
	}
	
	
	// Visualize the data
	/*IP::normalize( im_char, short(255) );
	ver.addObject( im_char, GLViewer::Volumn::MIP ); */
	ver.addObject( im_short, GLViewer::Volumn::MIP ); 
	ver.go(400, 200, 2);

	return 0;
}

