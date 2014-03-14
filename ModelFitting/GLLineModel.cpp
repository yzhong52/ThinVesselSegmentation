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


	WaitForSingleObject( hMutex, INFINITE );
	//glBegin( GL_POINTS );
	//for( int i=0; i < (int) dataPoints.size(); i++ ) {
	//	int lineID = labelings[i]; 
	//	Vec3f prj = lines[lineID].projection( dataPoints[i] ); 
	//	glColor3f( 1.0f, 0.0f, 0.0f ); 
	//	glVertex3fv( &prj[0] ); 
	//	glColor3f( 1.0f, 1.0f, 1.0f ); 
	//	glVertex3iv( &dataPoints[i][0] ); 
	//} 
	//glEnd();

	glBegin( GL_LINES );
	for( int i=0; i < (int) dataPoints.size(); i++ ) {
		int lineID = labelings[i]; 
		Vec3f prj = lines[lineID].projection( dataPoints[i] ); 
		glColor3f( 1.0f, 0.0f, 0.0f ); 
		glVertex3fv( &prj[0] ); 
		glColor3f( 1.0f, 1.0f, 1.0f ); 
		glVertex3iv( &dataPoints[i][0] ); 
	} 
	glEnd();

	ReleaseMutex( hMutex );
	
}


void GLLineModel::updatePoints( const vector<Vec3i>& pts ){
	WaitForSingleObject( hMutex, INFINITE );
	dataPoints = pts; 
	ReleaseMutex( hMutex );
}

void GLLineModel::updateModel( const vector<Line3D>& lns, const vector<int>& lbls )
{
	WaitForSingleObject( hMutex, INFINITE );
	if( lbls.size()==dataPoints.size() ) {
		lines = lns; 
		labelings = lbls; 
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
}