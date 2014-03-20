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

#include "Line3DTwoPoint.h" 

// for visualization
GLViwerModel ver;


// thead function for visualization 
void visualization_func( void* data ) {
	GLViwerModel& ver = *(GLViwerModel*) data; 
	ver.go(512, 300, 1);
}

const GC::EnergyType LOGLIKELIHOOD = 100; 


GC::EnergyType computeEnergy( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	GC::EnergyType energy = 0; 

	for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
		GC::LabelID label = labelings[site];
		
		const Line3D* line = lines[label];
		// distance from a point to a line
		const int& x = dataPoints[site][0];
		const int& y = dataPoints[site][1];
		const int& z = dataPoints[site][2];

		// log likelihood based on the distance
		GC::EnergyTermType loglikelihood = line->loglikelihood( Vec3f(1.0f * x,1.0f * y,1.0f * z) ); // dist * dist / ( 2 * line.sigma * line.sigma );

		energy += LOGLIKELIHOOD * loglikelihood; 
	}
	return energy; 
}


Mat computeEnergyMatrix( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	Mat eng( (int) dataPoints.size(), 1, CV_64F ); 

	for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
		GC::LabelID label = labelings[site];
		
		const Line3D* line = lines[label];
		// distance from a point to a line
		const int& x = dataPoints[site][0];
		const int& y = dataPoints[site][1];
		const int& z = dataPoints[site][2];
		// log likelihood based on the distance
		GC::EnergyTermType loglikelihood = line->loglikelihood( Vec3f(1.0f * x,1.0f * y,1.0f * z) );

		eng.at<double>( site, 0 ) = sqrt( LOGLIKELIHOOD * loglikelihood ); 
	}
	return eng; 
}

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
	/*for( int i=5; i<15; i++ ) {
		im_short.at(i,  i,  i)   = 100; 
		im_short.at(i,  i,  i+1) = 100; 
		im_short.at(i,  i+1,i)   = 100; 
		im_short.at(i+1,i,  i)   = 100; 
	}*/
	im_short.at(5, 5, 5) = 100; 
	im_short.at(5, 5, 10) = 100; 
	im_short.at(5, 5, 13) = 100; 
	im_short.at(10, 10, 10) = 100; 
	im_short.at(15, 15, 15) = 100; 

	// OR real data
	//im_short.load( "../data/data15.data" );

	// threshold the data and put the data points into a vector
	Data3D<unsigned char> im_uchar;
	vector<cv::Vec3i> dataPoints;
	IP::threshold( im_short, im_uchar, dataPoints, short(50) );
	
	// this is for visualization
	GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im_short.get_size() );
	ver.objs.push_back( model );

	//////////////////////////////////////////////////
	// create a thread for rendering
	//////////////////////////////////////////////////
	HANDLE thread_render = NULL; 
	thread_render = (HANDLE) _beginthread( visualization_func, 0, (void*)&ver ); 
	
	model->updatePoints( dataPoints ); 

	//////////////////////////////////////////////////
	// Line Fitting
	//////////////////////////////////////////////////
	// Initial Sampling - random
	const int num_init_labels = 2; 
	vector<Line3D*> lines; 

	Line3DTwoPoint* line = new Line3DTwoPoint();
	line->setPositions( Vec3f(7,7,5), Vec3f(3,3,15) ); 
	lines.push_back( line ); 

	Line3DTwoPoint* line2 = new Line3DTwoPoint();
	line2->setPositions( Vec3f(10,10,10), Vec3f(13,13,17) ); 
	lines.push_back( line2 ); 


	// Initial labelings
	vector<int> labelings = vector<int>( dataPoints.size(), 1 ); 
	model->updateModel( lines, labelings ); 

	cout << "Graph Cut Begin" << endl; 
	try{
		// keep track of energy in previous iteration
		GC::EnergyType energy_before = -1;

		for( int gciter=0; gciter<1; gciter++ ) { // TODO: let's run the algorithm for only one iteration for now
			// TODO: let's not have background model for now. We will add background model later
			GCoptimizationGeneralGraph gc( (int) dataPoints.size(), (int) lines.size() ); 

			////////////////////////////////////////////////
			// Data Costs
			////////////////////////////////////////////////
			for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
				GC::LabelID label;
				for( label = 0; label < lines.size(); label++ ){
					const Line3D* line = lines[label];
					// distance from a point to a line
					const int& x = dataPoints[site][0];
					const int& y = dataPoints[site][1];
					const int& z = dataPoints[site][2];
					// log likelihood based on the distance
					GC::EnergyTermType loglikelihood = line->loglikelihood( Vec3f(1.0f * x,1.0f * y,1.0f * z) );
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
			cout << "Iteration: " << gciter << ". Fitting Begin. Please Wait..."; 
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
			double lambda = 1e4; 
			for( int lmiter = 0; lambda < 10e100; lmiter++ ) { 
				cout << "Levenburg Maquart: " << lmiter << " Lambda: " << lambda << endl; 

				// there are six parameters
				// Jacobian Matrix ( # of cols: number of data points; # of rows: number of parameters of each line models)? 
				Mat Jacobian = Mat::zeros(
					(int) dataPoints.size(), 
					(int) lines.size() * lines[0]->getNumOfParameters(),
					CV_64F ); 
				
				// Contruct Jacobian matrix
				for( int label=0; label < lines.size(); label++ ) {
					for( int site=0; site < dataPoints.size(); site++ ) {
						if( labelings[site] != label ) {
							for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
								Jacobian.at<double>( site, 6*label+i ) = 0; 
							}
						} 
						else 
						{
							static const float delta = 0.001f; 

							// TODO: this line is not necessary if we have run graph cut
							energy_before = computeEnergy( dataPoints, labelings, lines ); 

							// compute the derivatives and construct Jacobian matrix
							for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
								lines[label]->updateParameterWithDelta( i, delta ); 
								Jacobian.at<double>( site, 6*label+i ) = 1.0 / delta * ( computeEnergy( dataPoints, labelings, lines ) - energy_before ); 
								lines[label]->updateParameterWithDelta( i, -delta ); 
							}
						}
					}
				} // end of contruction of Jacobian Matrix

				Mat A = Jacobian.t() * Jacobian; 
				
				A = A + Mat::diag( lambda * Mat::ones(A.cols, 1, CV_64F) ); 
				
				Mat B = Jacobian.t() * computeEnergyMatrix( dataPoints, labelings, lines ); 
				
				Mat X; 
				cv::solve( A, -B, X, DECOMP_QR  ); 
				for( int i=0; i<X.rows; i++ ) {
					std::cout << std::setw(14) << std::scientific << X.at<double>(i) << "  ";
				}
				cout << endl;

				for( int label=0; label < lines.size(); label++ ) {
					for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
						const double& delta = X.at<double>( label * lines[label]->getNumOfParameters() + i ); 
						lines[label]->updateParameterWithDelta( i, delta ); 
					}
				}

				double energyDiff = computeEnergy( dataPoints, labelings, lines ) - energy_before;
				if( energyDiff < 0 ) { // if energy is decreasing 
					cout << "-" << endl; 
					model->updateModel( lines, labelings ); 
					// the smaller lambda is, the faster it converges
					lambda *= 0.25; 
				} else {
					// If an iteration gives insufficient reduction in the residual, lamda can be increased, 
					// giving a step closer to the gradient descent direction 
					cout << "+" << endl; 
					for( int label=0; label < lines.size(); label++ ) {
						for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
							const double& delta = X.at<double>( label * lines[label]->getNumOfParameters() + i ); 
							lines[label]->updateParameterWithDelta( i, -delta ); 
						}
					}

					// the bigger lambda is, the slower it converges
					lambda *= 2.0; 
				}

				// Sleep(200);  // TODO: this is only for debuging 
				
			}
		}
	}
	catch (GCException e){
		e.Report();
	}
	
	cout << "Main Thread is Done. " << endl; 
	WaitForSingleObject( thread_render, INFINITE);

	for( int i=0; i<num_init_labels; i++ ){
		delete lines[i]; 
		lines[i] = NULL; 
	}

	return 0; 
}
