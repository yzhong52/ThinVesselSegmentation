#include "GLLineModel.h"

using namespace GLViewer;



GLLineModel::GLLineModel( cv::Vec3i size )
    : size( size ), render_mode(1)
{
    // creat a mutex
    // hMutex = CreateMutex( NULL, false, NULL );
}

GLLineModel::~GLLineModel( void )
{
    // destroy the mutex
    // CloseHandle( hMutex );
}


void normal_vectors(  const Vec3f v1, Vec3f& v2, Vec3f& v3 ) {
	// an arbitrary combination of the following two vectors
	// is a normal to v1
	// alpha * (-v1[1], v1[0], 0) + beta * (0, -v1[2], v[1])

	if( abs(v1[0])>1e-8 || abs(v1[1])>1e-8 ) {
		v2[0] = -v1[1];
		v2[1] =  v1[0];
		v2[2] =  0;
	} else {
		v2[0] =  0;
		v2[1] = -v1[2];
		v2[2] =  v1[1];
	}

    v3 = v2.cross( v1 );

    // normalize them so that v2*v2 + v3*v3 = 1
    const float length = sqrt( v2.dot( v2 ) + v3.dot( v3 ) );
	v2 /= length;
	v3 /= length;
}

void DrawCircle(const Vec3f& centre, const Vec3f& dir, const float& r )
{
    const int num_segments = 10;
    float theta = 2 * 3.1415926 / float(num_segments);
    float tangetial_factor = tanf(theta);//calculate the tangential factor

    float radial_factor = cosf(theta);//calculate the radial factor

    float x = r;//we start at angle = 0
    float y = 0;

    Vec3f norm1, norm2;
    normal_vectors( dir, norm1, norm2 );

    smart_assert( dir.dot(norm1)<1e-3, "Should be perpendicular with each other" );
    smart_assert( dir.dot(norm2)<1e-3, "Should be perpendicular with each other" );

    glBegin(GL_LINE_LOOP);
    for(int ii = 0; ii < num_segments; ii++)
    {
        Vec3f pos = centre + norm1 * x + norm2 * y;
        glVertex3fv( &pos[0] );//output vertex

        //calculate the tangential vector
        //remember, the radial vector is (x, y)
        //to get the tangential vector we flip those coordinates and negate one of them
        float tx = -y;
        float ty = x;

        //add the tangential vector
        x += tx * tangetial_factor;
        y += ty * tangetial_factor;

        //correct using the radial factor
        x *= radial_factor;
        y *= radial_factor;
    }
    glEnd();
}

void GLLineModel::render( void )
{
    // in case there is any previously bind texture, you need to unbind them

    /////////////////////////////////////////////////////
    ////// Draw the axis
    /////////////////////////////////////////////////////
    //glBegin( GL_LINES );
    //// x-axis
    //glColor3f(  1.0f, 0.0f, 0.0f );
    //glVertex3i( 0, 0, 0 );
    //glVertex3i( size[0], 0, 0 );
    //// y-axis
    //glColor3f(  0.0f, 1.0f, 0.0f );
    //glVertex3i( 0, 0, 0 );
    //glVertex3i( 0, size[1], 0 );
    //// z-axis
    //glColor3f(  0.0f, 0.0f, 1.0f );
    //glVertex3i( 0, 0, 0 );
    //glVertex3i( 0, 0, size[2] );
    //glEnd();

    // WaitForSingleObject( hMutex, INFINITE );

    /////////////////////////////////////////////////
    // draw the projection points
    /////////////////////////////////////////////////
    if( render_mode & 4 )
    {
        glPointSize( 3.0 );
        glBegin( GL_POINTS );
        for( int i=0; i < (int) dataPoints.size(); i++ )
        {
            const int& lineID = labelings[i];
            Vec3f prj = lines[lineID]->projection( dataPoints[i] );
            glColor3ubv( &lineColors[lineID][0] );
            glVertex3fv( &prj[0] );
        }
        glEnd();
    }

    /////////////////////////////////////////////////
    // draw the end points of the lines
    /////////////////////////////////////////////////
    //glPointSize( 6.0 );
    //glBegin( GL_POINTS );
    //for( int i=0; i < (int) dataPoints.size(); i++ ) {
    //	int lineID = labelings[i];
    //	Vec3f p1, p2;
    //	lines[lineID]->getEndPoints( p1, p2 );
    //	glColor3ubv( &lineColors[lineID][0] );
    //	glVertex3fv( &p1[0] );
    //	glVertex3fv( &p2[0] );
    //}
    //glEnd();

    /////////////////////////////////////////////////
    // draw a short line alond the line model
    /////////////////////////////////////////////////
    if( render_mode & 1 )
    {
        glColor3f( 0.4f, 0.4f, 0.4f );
        glBegin( GL_LINES );
        for( int i=0; i < (int) dataPoints.size(); i++ )
        {
            const int lineID = labelings[i]; // label id
            const Vec3f prj = lines[lineID]->projection( dataPoints[i] );  // position
            const Vec3f dir = lines[lineID]->getDirection(); // direction
            glVertex3fv( &(prj + dir * 0.5 )[0] );
            glVertex3fv( &(prj - dir * 0.5 )[0] );
        }
        glEnd();
    }

    if( render_mode & 2 )
    {
        /////////////////////////////////////////////////
        // draw the lines of the projection direction
        /////////////////////////////////////////////////
        glBegin( GL_LINES );
        for( int i=0; i < (int) dataPoints.size(); i++ )
        {
            const int lineID = labelings[i]; // label id
            Vec3f prj = lines[lineID]->projection( dataPoints[i] ); // position
            // draw projection point
            glColor3ubv( &lineColors[lineID][0] );
            glVertex3fv( &prj[0] );
            // draw data points
            glColor3f( 0.1f, 0.1f, 0.1f );
            glVertex3iv( &dataPoints[i][0] );
        }
        glEnd();
    }

    if( render_mode & 8 )
    {
        for( int i=0; i < (int) dataPoints.size(); i++ ){
            const int lineID = labelings[i]; // label id
            const Vec3f prj = lines[lineID]->projection( dataPoints[i] ); // position
            const Vec3f dir = lines[lineID]->getDirection();

            glColor3ubv( &lineColors[lineID][0] );
            DrawCircle( prj, dir, lines[lineID]->getSigma() );
        }

    }
    // ReleaseMutex( hMutex );
}


void GLLineModel::updatePoints( const vector<Vec3i>& pts )
{
    // WaitForSingleObject( hMutex, INFINITE );
    dataPoints = pts;
    // ReleaseMutex( hMutex );
}

void GLLineModel::updateModel( const vector<Line3D*>& lns, const vector<int>& lbls )
{
    // WaitForSingleObject( hMutex, INFINITE );
    if( lbls.size()==dataPoints.size() )
    {
        lines = lns;
        labelings = lbls;
        int num = (int) lns.size() - (int) lineColors.size();
        for( int i=0; i<num; i++ )
        {
            Vec3b c(
                (rand()%228 ) + 28,
                (rand()%228 ) + 28,
                (rand()%228 ) + 28 );
            lineColors.push_back( c );
        }
    }
    else
    {
        cout << "Error: Update model fail." << endl;
        std::cout << "  Location: file "<< __FILE__ << ", line " << __LINE__ << std::endl;
        system( "pause" );
    }
    // ReleaseMutex( hMutex );
}

void GLLineModel::init(void)
{
    //glDisable(GL_DEPTH_TEST);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_ONE, GL_ONE);
    //glBlendEquation( GL_MAX_EXT );
    //cout << "Volumn Rendeing Mode is set to MIP" << endl;

    //// Antialiasing
    //glDisable(GL_LINE_SMOOTH);
    //glHint (GL_LINE_SMOOTH_HINT, GL_FASTEST );

    //glEnable( GL_POLYGON_SMOOTH_HINT );
    //glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    //glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    //glDisable(GL_DEPTH_TEST);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_ONE, GL_ONE);
    //glBlendEquation( GL_MAX_EXT );

    // glEnable( GL_POINT_SPRITE ); // GL_POINT_SPRITE_ARB if you're
    // using the functionality as an extension.

    //glEnable( GL_POINT_SMOOTH );
    //glEnable( GL_BLEND );
    //glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

}

void GLLineModel::keyboard( unsigned char key )
{
    render_mode++;
    if( render_mode==0 ) render_mode = 1;
}
