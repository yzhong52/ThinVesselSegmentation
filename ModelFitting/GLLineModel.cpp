#include "GLLineModel.h"

using namespace GLViewer;



GLLineModel::GLLineModel( cv::Vec3i size ) : size( size ) {
	// creat a mutex 
	hMutex = CreateMutex( NULL, false, NULL );
}

GLLineModel::~GLLineModel( void ){
	// destroy the mutex 
	CloseHandle( hMutex ); 
}


void GLLineModel::render(void){
	// in case there is any previously bind texture, you need to unbind them
	glBindTexture( GL_TEXTURE_3D, NULL );

	// Also draw the axis
	glBegin( GL_LINES );
	// x-axis
	glColor3f(  1.0f, 0.0f, 0.0f ); 
	glVertex3i( 0, 0, 0 ); 
	glVertex3i( size[0], 0, 0 ); 
	// y-axis
	glColor3f(  0.0f, 1.0f, 0.0f ); 
	glVertex3i( 0, 0, 0 ); 
	glVertex3i( 0, size[1], 0 ); 
	// z-axis
	glColor3f(  0.0f, 0.0f, 1.0f ); 
	glVertex3i( 0, 0, 0 ); 
	glVertex3i( 0, 0, size[2] ); 
	glEnd();

	WaitForSingleObject( hMutex, INFINITE );

	// draw the data points 
	glBegin( GL_POINTS );
	for( int i=0; i < (int) dataPoints.size(); i++ ) {
		int lineID = labelings[i]; 
		// actual position
		Vec3f prj = lines[lineID]->projection( dataPoints[i] ); 
		glColor3ubv( &lineColors[lineID][0] ); 
		glVertex3fv( &prj[0] ); 
		// data points
		glColor3f( 0.1f, 0.1f, 0.1f ); 
		glVertex3iv( &dataPoints[i][0] ); 
	} 
	glEnd();

	glBegin( GL_LINES );
	for( int i=0; i < (int) dataPoints.size(); i++ ) {
		int lineID = labelings[i]; 
		// actual position
		Vec3f prj = lines[lineID]->projection( dataPoints[i] ); 
		glColor3ubv( &lineColors[lineID][0] ); 
		glVertex3fv( &prj[0] ); 
		// data points
		glColor3f( 0.1f, 0.1f, 0.1f ); 
		glVertex3iv( &dataPoints[i][0] ); 
	} 
	glEnd();

	// draw the models
	//glBegin( GL_LINES );
	//for( int i=0; i < (int) lines.size(); i++ ) {
	//	glColor3fv( &lineColors[i][0] ); 
	//	Vec3i p1 = lines[i].getPos(); 
	//	Vec3i p2 = 2 * lines[i].getDir() + lines[i].getPos(); 
	//	glVertex3iv( &p1[0] ); 
	//	glVertex3iv( &p2[0] ); 
	//} 
	//glEnd();

	ReleaseMutex( hMutex );
	
}


void GLLineModel::updatePoints( const vector<Vec3i>& pts ){
	WaitForSingleObject( hMutex, INFINITE );
	dataPoints = pts; 
	ReleaseMutex( hMutex );
}

void GLLineModel::updateModel( const vector<Line3D*>& lns, const vector<int>& lbls )
{
	WaitForSingleObject( hMutex, INFINITE );
	if( lbls.size()==dataPoints.size() ) {
		lines = lns; 
		labelings = lbls; 
		int num = (int) lns.size() - (int) lineColors.size();
		for( int i=0; i<num; i++ ) {
			Vec3b c( 
				(rand()%228 ) + 28,
				(rand()%228 ) + 28,
				(rand()%228 ) + 28 ); 
			lineColors.push_back( c ); 
		}
	} else {
		cout << "Error: Update model fail." << endl; 
		std::wcout << "  Location: file "<< _CRT_WIDE(__FILE__) << ", line " << __LINE__ << std::endl; 
		system( "pause" );
	}
	ReleaseMutex( hMutex );
}

void GLLineModel::init(void){
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation( GL_MAX_EXT ); 
	//cout << "Volumn Rendeing Mode is set to MIP" << endl;

	//// Antialiasing
	glDisable(GL_LINE_SMOOTH);
	glHint (GL_LINE_SMOOTH_HINT, GL_FASTEST );

	//glEnable( GL_POLYGON_SMOOTH_HINT );
	//glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	//glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//glDisable(GL_DEPTH_TEST);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_ONE, GL_ONE);
	//glBlendEquation( GL_MAX_EXT ); 

	
	
	glEnable( GL_POINT_SPRITE ); // GL_POINT_SPRITE_ARB if you're
                                 // using the functionality as an extension.

    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 3.0 );
}