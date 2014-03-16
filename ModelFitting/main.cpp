// ModelFitting.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <Windows.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "Line3D.h"

// For multithreading
#include <Windows.h>
#include <process.h>
#include <iomanip>

// This project is build after VesselNess. 
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess. 
#include "Data3D.h" 
#include "GLViwerModel.h"
#include "MinSpanTree.h" 

// For the use of graph cut
#include "GCoptimization.h"
typedef GCoptimization GC; 

// for visualization
GLViwerModel ver;

void visualization_func( void* data ) {
	GLViwerModel& ver = *(GLViwerModel*) data; 
	ver.go();
}

const GC::EnergyType LOGLIKELIHOOD = 100; 


GC::EnergyType computeEnergy( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D>& lines )
{
	GC::EnergyType energy = 0; 

	for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
		GC::LabelID label = labelings[site];
		
		const Line3D& line = lines[label];
		// distance from a point to a line
		const int& x = dataPoints[site][0];
		const int& y = dataPoints[site][1];
		const int& z = dataPoints[site][2];
		float dist = line.distanceToLine( Vec3f(1.0f * x,1.0f * y,1.0f * z) );
		// log likelihood based on the distance
		GC::EnergyTermType loglikelihood = dist * dist / ( 2 * line.sigma * line.sigma );

		energy += LOGLIKELIHOOD * loglikelihood; 
	}
	return energy; 
}


Mat computeEnergyMatrix( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D>& lines )
{
	Mat eng( (int) dataPoints.size(), 1, CV_64F ); 

	for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
		GC::LabelID label = labelings[site];
		
		const Line3D& line = lines[label];
		// distance from a point to a line
		const int& x = dataPoints[site][0];
		const int& y = dataPoints[site][1];
		const int& z = dataPoints[site][2];
		float dist = line.distanceToLine( Vec3f(1.0f * x,1.0f * y,1.0f * z) );
		// log likelihood based on the distance
		GC::EnergyTermType loglikelihood = dist * dist / ( 2 * line.sigma * line.sigma );

		eng.at<double>( site, 0 ) = sqrt( LOGLIKELIHOOD * loglikelihood ); 
	}
	return eng; 
}

// TODO: Fix this function.
int main(int argc, char* argv[])
{

	//std::cout.width(1);
	//for( int i=0; i<10; i++ ){
	//	cout.width( 2 ); 
	//	std::cout << std::setw(14) << std::scientific << 1.343434343435e3 << "  ";
	//	std::cout << std::setw(14) << std::scientific << -1.343434343435e3 << "  ";
	//}
	////Mat Jacobian = Mat::ones( 3, 4, CV_32F ); 
	////cout << Jacobian << endl;
	////Jacobian.at<float>(2,3) = 2; 
	////cout << Jacobian << endl;
	//return 0; 

	CreateDirectory(L"./output", NULL);
	
	Data3D<short> im_short;
	//Synthesic Data
	im_short.reset( Vec3i(20,20,20) ); 
	for( int i=5; i<15; i++ ) {
		im_short.at(i,  i,  i)   = 100; 
		im_short.at(i,  i,  i+1) = 100; 
		im_short.at(i,  i+1,i)   = 100; 
		im_short.at(i+1,i,  i)   = 100; 
	}
	// OR real data
	//im_short.load( "../data/data15.data" );

	// threshold the data and put the data points into a vector
	Data3D<unsigned char> im_uchar;
	vector<cv::Vec3i> dataPoints;
	IP::threshold( im_short, im_uchar, dataPoints, short(50) );

	GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im_short.get_size() );
	ver.objs.push_back( model );

	//////////////////////////////////////////////////
	// create a thread for rendering
	//////////////////////////////////////////////////
	HANDLE thread_render = (HANDLE) _beginthread( visualization_func, 0, (void*)&ver ); 
	
	model->updatePoints( dataPoints ); 

	//////////////////////////////////////////////////
	// Line Fitting
	//////////////////////////////////////////////////
	// Initial Sampling - random
	const int num_init_labels = 2; 
	vector<Line3D> lines; // three float of geometric locations and three for directions
	for( int i=0; i<num_init_labels; i++ ){
		Line3D line;
		// position of the line
		int pointID = rand() % dataPoints.size(); 
		line.setPos( dataPoints[pointID] ); 
		// direction of the line 
		line.setDir( Vec3f(
			(float) rand()+1, 
			(float) rand()+1, 
			(float) rand()+1 )); 
		lines.push_back( line );
	}

	vector<int> labelings = vector<int>( dataPoints.size(), 0 ); 

	model->updateModel( lines, labelings ); 

	cout << "Graph Cut Begin" << endl; 
	try{
		// keep track of energy in previous iteration
		GC::EnergyType energy_before = -1;

		for( int i=0; i<1; i++ ) { // TODO: let's run the algorithm for only one iteration
			// TODO: let's not have background model for now. We will add background model later
			GCoptimizationGeneralGraph gc( (int) dataPoints.size(), (int) lines.size() ); 

			////////////////////////////////////////////////
			// Data Costs
			////////////////////////////////////////////////
			for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
				GC::LabelID label;
				for( label = 0; label < lines.size(); label++ ){
					const Line3D& line = lines[label];
					// distance from a point to a line
					const int& x = dataPoints[site][0];
					const int& y = dataPoints[site][1];
					const int& z = dataPoints[site][2];
					float dist = line.distanceToLine( Vec3f(1.0f * x,1.0f * y,1.0f * z) );
					// log likelihood based on the distance
					GC::EnergyTermType loglikelihood = dist * dist / ( 2 * line.sigma * line.sigma );
					// loglikelihood += log( line.sigma ); 
					// static double C = 0.5 * log( 2*M_PI );
					// loglikelihood += C; 
					gc.setDataCost( site, label, LOGLIKELIHOOD * loglikelihood );
				}
			}

			////////////////////////////////////////////////
			// Smooth Cost
			////////////////////////////////////////////////
			// ... TODO: Setting Smooth Cost
			// im_uchar
			

			////////////////////////////////////////////////
			// Graph-Cut Begin
			////////////////////////////////////////////////
			cout << "Iteration: " << i << ". Fitting Begin. Please Wait..."; 
			gc.expansion(1); // run expansion for 1 iterations. For swap use gc->swap(num_iterations);
			GC::EnergyType cur_energy = gc.compute_energy();
			if ( energy_before==cur_energy ) { 
				cout << endl << "Energy is not changing. " << endl; break; 
			} else {
				energy_before = cur_energy; 
			}
			cout << "Done. " << endl;
			
			// Counting the number of labels in forground and background 
			for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
				const int& x = dataPoints[site][0];
				const int& y = dataPoints[site][1];
				const int& z = dataPoints[site][2];
				labelings[site] = gc.whatLabel( site ); 
			}
			
			model->updateModel( lines, labelings ); 

			////////////////////////////////////////////////
			// Re-estimation
			////////////////////////////////////////////////
			// Levenburg Maquart
			double lambda = 1e2; 
			for( int lmiter = 0; lambda < 10e10; lmiter++ ) { 
				cout << "Levenburg Maquart: " << lmiter << " Lambda: " << lambda << endl; 

				// there are six parameters
				// Jacobian Matrix ( # of cols: number of data points; # of rows: number of parameters of each line models)? 
				Mat Jacobian = Mat::zeros(
					(int) dataPoints.size(), 
					(int) lines.size() * 6,
					CV_64F ); 
				
				// Contruct Jacobian matrix
				for( int label=0; label < lines.size(); label++ ) {
					for( int site=0; site < dataPoints.size(); site++ ) {
						if( labelings[site] != label ) {
							Jacobian.at<double>( site, 6*label ) = 0; 
							Jacobian.at<double>( site, 6*label+1 ) = 0; 
							Jacobian.at<double>( site, 6*label+2 ) = 0; 
							Jacobian.at<double>( site, 6*label+3 ) = 0; 
							Jacobian.at<double>( site, 6*label+4 ) = 0; 
							Jacobian.at<double>( site, 6*label+5 ) = 0; 
						} else {
							static const float delta = 0.01f; 
							for( int i=0; i<3; i++ ) {
								lines[label].pos[i] += delta; 
								Jacobian.at<double>( site, 6*label+i ) = 
									1.0 / delta * ( computeEnergy( dataPoints, labelings, lines ) - energy_before ); 
								lines[label].pos[i] -= delta; 
							}
							for( int i=0; i<3; i++ ) {
								lines[label].dir[i] += delta; 
								Jacobian.at<double>( site, 6*label+i+3 ) = 
									1.0f / delta * ( computeEnergy( dataPoints, labelings, lines ) - energy_before ); 
								lines[label].dir[i] -= delta; 
							}
						}
					}
				}
				// end of contruction of Jacobian Matrix
				Mat A = Jacobian.t() * Jacobian; 
				
				A = A + Mat::diag( lambda * Mat::ones(A.cols, 1, CV_64F) ); 
			
				Mat B = Jacobian.t() * computeEnergyMatrix( dataPoints, labelings, lines ); 
			
				Mat X; 
				cv::solve( A, -B, X ); 
			
				
				for( int i=0; i<X.rows; i++ ) {
					std::cout << std::setw(14) << std::scientific << X.at<double>( i ) << "  ";
				}
				cout << endl;
				
				for( int i=0; i<lines.size(); i++ ) {
					lines[i].pos[0] += (float) X.at<double>( 6*i ); 
					lines[i].pos[1] += (float) X.at<double>( 6*i+1 ); 
					lines[i].pos[2] += (float) X.at<double>( 6*i+2 ); 

					lines[i].dir[0] += (float) X.at<double>( 6*i+3 ); 
					lines[i].dir[1] += (float) X.at<double>( 6*i+4 ); 
					lines[i].dir[2] += (float) X.at<double>( 6*i+5 ); 
				}
				double energyDiff = computeEnergy( dataPoints, labelings, lines ) - energy_before;
				if( energyDiff < 0 ) { // if energy is decreasing 
					model->updateModel( lines, labelings ); 
					lambda *= 2; 
				} else {
					for( int i=0; i<lines.size(); i++ ) {
						lines[i].pos[0] -= (float) X.at<double>( 6*i ); 
						lines[i].pos[1] -= (float) X.at<double>( 6*i+1 ); 
						lines[i].pos[2] -= (float) X.at<double>( 6*i+2 ); 

						lines[i].dir[0] -= (float) X.at<double>( 6*i+3 ); 
						lines[i].dir[1] -= (float) X.at<double>( 6*i+4 ); 
						lines[i].dir[2] -= (float) X.at<double>( 6*i+5 ); 
					}
					lambda /= 2; 
				}
			}
		}
	}
	catch (GCException e){
		e.Report();
	}
	
	cout << "Main Thread is Done. " << endl; 
	WaitForSingleObject( thread_render, INFINITE);
	return 0; 
}
