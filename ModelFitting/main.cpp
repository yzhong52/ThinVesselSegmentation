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


bool load_graph( vector<Line3D>& lines, vector<vector<Vec3i> >& pointsSet, cv::Vec3i& size, const std::string& file_name ) {
	size = cv::Vec3i(0,0,0);

	/////////////////////////////////////////////////////////////
	// Loading Data
	/////////////////////////////////////////////////////////////
	// Yuchen: I am working with my collegue Xuefeng. 
	// The data are supposed to be a bunch of line segments in 3D
	// Open File
	std::ifstream fin1, fin2;
	string filename1 = file_name;
	string filename2 = file_name;
	if( file_name.back()!='/' ) {
		filename1 += ".";
		filename2 += ".";
	}
	filename1 += "linedata.txt";
	filename2 += "models.txt";
	
	
	fin1.open( filename1 );
	fin2.open( filename2 );
	if( !fin1.is_open() ){
		std::cerr << "File cannot be open" << endl;
		return 0;
	}
	if( !fin2.is_open() ){
		std::cerr << "File cannot be open" << endl;
		return 0;
	}

	int num_line1, num_line2;
	// Reading Data from file
	fin1 >> num_line1;
	fin2 >> num_line2;
	if( num_line1 != num_line2 ){
		cout << "Data Does not match" << endl;
		return 0;
	}
	int& num_line = num_line1;
	
	for( int i=0; i<num_line1; i++ ) {
		Line3D line;
		// each line segments are defined as two end points in 3d
		fin1 >> line.pos[0]; 
		fin1 >> line.pos[1]; 
		fin1 >> line.pos[2]; 
		fin1 >> line.dir[0]; 
		fin1 >> line.dir[1]; 
		fin1 >> line.dir[2]; 
		line.dir[0] -= line.pos[0]; 
		line.dir[1] -= line.pos[1]; 
		line.dir[2] -= line.pos[2]; 
		float len = sqrt( line.dir.dot( line.dir ) );
		line.dir[0] /= len;
		line.dir[1] /= len;
		line.dir[2] /= len;

		float temp;
		fin2 >> temp; if( temp!= line.pos[0] ) { cout << "Error: Data does not match" << endl; return 0; };
		fin2 >> temp;
		fin2 >> temp;
		fin2 >> temp;
		fin2 >> temp;
		fin2 >> temp;
		fin2 >> line.sigma;

		// There are cooresponding points that are assigned to this label (or line)
		vector<Vec3i> points;
		int num_points;
		fin1 >> num_points;
		for( int j=0; j<num_points; j++ ) {
			Vec3i p;
			fin1 >> p[0];
			fin1 >> p[1];
			fin1 >> p[2];
			size[0] = max( p[0], size[0] );
			size[1] = max( p[1], size[1] );
			size[2] = max( p[2], size[2] );
			points.push_back( p );
		}
		pointsSet.push_back( points );
		lines.push_back( line );
	}
	fin1.close();
	fin2.close();
	size[0]++; size[1]++; size[2]++; 
	return true; 
}


void visualize_ryen_model_fitting_result(void){
	// loading the data
	vector<Line3D> lines;
	vector<vector<Vec3i> > pointsSet;
	cv::Vec3i size;
	load_graph( lines, pointsSet, size, "data/ROI19/" );
	// visualizing the data
	ver.addModel( lines, pointsSet, size );
	
	//MST::Graph< MST::Edge_Ext, MST::LineSegment > line_tree;
	//MinSpanTree::build_tree_xuefeng( "data/ROI_15_Fast_search_1/", line_tree, 250 );
	//ver.addObject( line_tree );

	Data3D<short> im_short;
	im_short.load( "../VesselNess/output/parts/vessel3d.rd.19.part7.data" );
	ver.addObject( im_short, GLViewer::Volumn::MIP ); 

	ver.go(400, 200, 2);
}

// TODO: Fix this function.
int main(int argc, char* argv[])
{
	visualize_ryen_model_fitting_result();
	return 0; 






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

