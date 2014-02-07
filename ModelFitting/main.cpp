// ModelFitting.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// This project is build after VesselNess. 
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess. 

#include "Data3D.h" 
#include "GLViwerWrapper.h"
#include <Windows.h>

#include "GCoptimization.h"
typedef GCoptimization GC; 

GLViewerExt ver;

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
	const int num_init_labels = 1; 
	vector<Vec6f> lines; // three float of geometric locations and three for directions
	for( int i=0; i<num_init_labels; i++ ){
		lines.push_back( Vec6f(1,1,1,1,1,1) );
	}
	
	
	// set up label cost
	// int* label_costs = new int[_num_labels+1];/*Add 1 for BG label*/
	try{
		GC::EnergyType energy_before = -1;

		for( int i=0; i<150; i++ ) { 
			GCoptimizationGeneralGraph gc( im_short.SX()*im_short.SY()*im_short.SZ(), (int)lines.size()+1 );
			
			// data costs
			for( int x=0; x<im_short.SX(); x++ ) {
				for( int y=0; y<im_short.SY(); y++ ) {
					for( int z=0; z<im_short.SZ(); z++ ) {
						GC::SiteID site = x + y * im_short.SX() + z * im_short.SY() * im_short.SX();
						GC::SiteID label;
						GC::EnergyTermType datacost = 4000-im_short.at(x,y,z); 
						// data cost to general label
						for( label = 0; label < lines.size(); label++ ){
							gc.setDataCost( site, label, datacost );
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
		}
	}
	catch (GCException e){
		e.Report();
	}
	
	
	// Visualize the data
	IP::normalize( im_char, short(255) );
	ver.addObject( im_char, GLViewer::Volumn::MIP ); 
	ver.addObject( im_short, GLViewer::Volumn::MIP ); 
	ver.go(400, 200, 2);

	return 0;
}

