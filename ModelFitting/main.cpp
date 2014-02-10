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

// For the use of graph cut
#include "GCoptimization.h"
typedef GCoptimization GC; 

GLViwerModel ver;

// TODO: Fix this function.
int main(int argc, char* argv[])
{
	CreateDirectory(L"./output", NULL);
	
	Data3D<short> im_short;
	im_short.load( "data/roi15.data" );
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
		line.pos[0] = float( rand() % im_short.SX() ); 
		line.pos[1] = float( rand() % im_short.SY() ); 
		line.pos[2] = float( rand() % im_short.SZ() ); 
		//line.pos[0] = 0; 
		//line.pos[1] = 0; 
		//line.pos[2] = 0; 
		// direction of the line 
		line.dir[0] = float( rand()+1 );
		line.dir[1] = float( rand()+1 );
		line.dir[2] = float( rand()+1 );
		//line.dir[0] = 1.0f * im_short.SX(); 
		//line.dir[1] = 1.0f * im_short.SY(); 
		//line.dir[2] = 1.0f * im_short.SZ(); 
		// normalize the direction
		float length = sqrt( line.dir.dot( line.dir ) ); 
		line.dir /= length; 
		lines.push_back( line );
	}
	
	// set up label cost
	// int* label_costs = new int[_num_labels+1];/*Add 1 for BG label*/
	try{
		GC::EnergyType energy_before = -1;

		for( int i=0; i<150; i++ ) { 
			GCoptimizationGeneralGraph gc( im_short.SX()*im_short.SY()*im_short.SZ(), (int)lines.size()+1 );
			
			// data costs
			for( int z=0; z<im_short.SZ(); z++ ) {
				for( int y=0; y<im_short.SY(); y++ ) {
					for( int x=0; x<im_short.SX(); x++ ) {		
						GC::SiteID site = x + y * im_short.SX() + z * im_short.SY() * im_short.SX();
						GC::SiteID label;
						GC::EnergyTermType datacost = 3500-im_short.at(x,y,z); // TODO make this a parameter to tune
						// data cost to general label
						for( label = 0; label < lines.size(); label++ ){
							const Line3D& line = lines[label];
							// distance from a point to a line
							float dist = line.distanceToLine( Vec3f(1.0f*x,1.0f*y,1.0f*z) );
							GC::EnergyTermType loglikelihood = dist*dist/( 2*line.sigma*line.sigma );
							loglikelihood += log( line.sigma ); 
							static float C = 0.5f * (float) log( 2*M_PI );
							loglikelihood += C; 
							loglikelihood *= 100; // TODO make this a parameter to tune
							gc.setDataCost( site, label, datacost + loglikelihood );
						}
						// data cost to background/outlier label
						gc.setDataCost( site, label, -datacost );
					}
				}
			}
			
			cout << "Iteration: " << i << ". Fitting Begin. Please Wait..."; 
			gc.expansion(1); // run expansion for 1 iterations. For swap use gc->swap(num_iterations);
			GC::EnergyType cur_energy = gc.compute_energy();
			if ( energy_before==cur_energy ) { 
				cout << endl << "Energy not changing. " << endl; break; 
			}
			else {
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

			ver.addModel( &gc, lines, im_short.get_size() );
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

