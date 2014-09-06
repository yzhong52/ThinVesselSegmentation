#include "GLVolumn.h"

using namespace std;

namespace GLViewer
{

Volumn::Volumn( unsigned char* im_data,
                const int& im_x, const int& im_y, const int& im_z,
                GLCamera* ptrCamera, float s )
    : sx( im_x ), sy( im_y ), sz( im_z ), ptrCam( ptrCamera ), scale( s )
{
    /* From wikipedia: Do not forget that all 3 dimensions must be a power
       of 2! so your 2D textures must have the same size and this size be
       a power of 2 AND the number of layers (=2D textures) you use to create
       your 3D texture must be a power of 2 too. */
    static const double log2 = log(2.0);
    texture_sx = (int) pow(2, ceil( log( 1.0*sx )/log2 ));
    texture_sy = (int) pow(2, ceil( log( 1.0*sy )/log2 ));
    texture_sz = (int) pow(2, ceil( log( 1.0*sz )/log2 ));

    // allocating memeory for texture
    int texture_size = texture_sx * texture_sy * texture_sz;
    data = new (std::nothrow) unsigned char [ texture_size ];
    if( !data )
    {
        cout << "Unable to allocate memory for OpenGL texture" << endl;
        return;
    }

    memset( data, 0, sizeof(unsigned char) * texture_size );
    for( int z=0; z<sz; z++ )
    {
        for( int y=0; y<sy; y++ )
        {
            for( int x=0; x<sx; x++ )
            {
                data[ z*texture_sy*texture_sx + y*texture_sx + x]
                    = im_data[ z*sy*sx + y*sx + x];
            }
        }
    }

    render_mode = MIP;
}


Volumn::~Volumn()
{
    if(data)
    {
        delete[] data;
        data = nullptr;
    }
}


void Volumn::setRenderMode( RenderMode mode )
{
    switch( mode )
    {
    case MIP:
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glBlendEquation( GL_MAX_EXT );
        cout << "Volumn Rendeing Mode is set to MIP" << endl;
        break;
    case CrossSection:
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        cout << "Volumn Rendeing Mode is set to CrossSection" << endl;
        break;
    case Surface:
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        cout << "Volumn Rendeing Mode is set to Surface" << endl;
        break;
    }
    render_mode = mode;
}



std::vector<cv::Vec3f> Volumn::intersectPoints( const cv::Vec3f& center,
        const cv::Vec3f& norm )
{
    float t;
    std::vector<cv::Vec3f> result;
    if( std::abs(norm[2]) > 1.0e-3 )
    {
        // (0, 0, t)
        t = center.dot(norm);
        t /= norm[2];
        if( t>=0 && t<=sz ) result.push_back( cv::Vec3f(0,0,t) );
        // (0, sy, t)
        t = center.dot(norm)- norm[1] * (float)sy;
        t /= norm[2];
        if( t>=0 && t<=sz ) result.push_back( cv::Vec3f(0,(float)sy,t) );
        // (sx, 0, t)
        t = center.dot(norm) - norm[0] * (float)sx;
        t /= norm[2];
        if( t>=0 && t<=sz ) result.push_back( cv::Vec3f((float)sx,0,t) );
        // (sx, sy, t)
        t = center.dot(norm) - norm[1] * (float)sy - norm[0] * (float)sx;
        t /= norm[2];
        if( t>=0 && t<=sz ) result.push_back( cv::Vec3f((float)sx,(float)sy,t) );
    }

    if( std::abs(norm[1]) > 1.0e-3 )
    {
        // (0, t, 0)
        t = center.dot(norm);
        t /= norm[1];
        if( t>=0 && t<=sy ) result.push_back( cv::Vec3f(0,t,0) );
        // (sx, t, 0)
        t = center.dot(norm) - norm[0] * (float)sx;
        t /= norm[1];
        if( t>=0 && t<=sy ) result.push_back( cv::Vec3f((float)sx,t,0) );
        // (0, t, sz)
        t = center.dot(norm) - norm[2] * (float)sz;
        t /= norm[1];
        if( t>=0 && t<=sy ) result.push_back( cv::Vec3f(0,t,(float)sz) );
        // (sx, t, sz)
        t = center.dot(norm) - norm[2] * (float)sz - norm[0] * (float)sx;
        t /= norm[1];
        if( t>=0 && t<=sy ) result.push_back( cv::Vec3f((float)sx,t,(float)sz) );
    }

    if( std::abs(norm[0]) > 1.0e-3 )
    {
        // (t, 0, 0)
        t = center.dot(norm);
        t /= norm[0];
        if( t>=0 && t<=sx ) result.push_back( cv::Vec3f(t,0,0) );
        // (t, sy, 0)
        t = center.dot(norm) - norm[1] * (float)sy;
        t /= norm[0];
        if( t>=0 && t<=sx ) result.push_back( cv::Vec3f(t,(float)sy,0) );
        // (t, 0, sz)
        t = center.dot(norm) - norm[2] * (float)sz;
        t /= norm[0];
        if( t>=0 && t<=sx ) result.push_back( cv::Vec3f(t,0,(float)sz) );
        // (t, sy, sz)
        t = center.dot(norm) - norm[1] * (float)sy - norm[2] * (float)sz;
        t /= norm[0];
        if( t>=0 && t<=sx ) result.push_back( cv::Vec3f(t,(float)sy,(float)sz) );
    }


    if( result.size()<=2 )
    {
        result.clear();
    }
    else if( result.size()==3 )
    {
        // This is a triangle. We don't need to do anything.
    }
    else if( result.size()<=6 )
    {
        // Sort them based on signed angle:
        // http://stackoverflow.com/questions/20387282/compute-the-cross-section-of-a-cube

        cv::Vec3f centroid(0,0,0);
        for( unsigned int i=0; i<result.size(); i++ )
        {
            centroid += result[i];
        }
        centroid /= (float) result.size();

        // We are not using the first index
        static float signed_angle[6];
        for( unsigned int i=0; i<result.size(); i++ )
        {
            static cv::Vec3f va[6];
            va[i] = result[i] - centroid;
            float length_vai = std::sqrt( va[i].dot( va[i] ) );
            float length_va0 = std::sqrt( va[0].dot( va[0] ) );
            float dotproduct = va[0].dot( va[i] )/( length_vai*length_va0 );
            // constraint the result of dotproduct be within -1 and 1 (it might
            // sometime not with this range only because of floating point
            // calculation accuracy )
            if( dotproduct<-1 )
            {
                dotproduct = -1;
            }
            else if( dotproduct>1 )
            {
                dotproduct = 1;
            }
            signed_angle[i] = (float)acos( dotproduct );
            if( std::abs( signed_angle[i] ) < 1e-3 ) continue;

            cv::Vec3f cross = va[0].cross( va[i] );
            if( cross.dot( norm ) < 0 )
            {
                signed_angle[i] = -signed_angle[i];
            }
        }
        // bubble sort the result by signed_angle
        for( unsigned int i=0; i<result.size(); i++ )
        {
            for( unsigned int j=i+1; j<result.size(); j++ )
            {
                if( signed_angle[i] < signed_angle[j] )
                {
                    std::swap( signed_angle[i], signed_angle[j] );
                    std::swap( result[i], result[j] );
                }
            }
        }
    }
    else
    {
        std::cout << "Error (Volumn.h): There are at most six points" << std::endl;
    }
    return result;
}


void Volumn::init(void)
{
    // Creating Textures
    glGenTextures(1, &texture); // Create The Texture
    glBindTexture(GL_TEXTURE_3D, texture);

    // Yuchen [Important]: For the following line of code
    /* If the graphic hard do not have enough memory for the 3D texture,
       OpenGL will fail to render the textures. However, since it is hardware
       related, the porgramm may not show any assertions here (this need to
       be double-checked later. But now, it has no problem rendering 3D
       texture with a size of 1024 * 1024 * 1024. */
    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE,
                 texture_sx, texture_sy, texture_sz, 0,
                 GL_LUMINANCE, GL_UNSIGNED_BYTE, data );

    //////////////////////////////////////
    // Set up OpenGL

    // Enable Blending For Maximum Intensity Projection
    setRenderMode( render_mode );

    // Antialiasing
    glEnable (GL_LINE_SMOOTH);
    glHint (GL_LINE_SMOOTH_HINT, GL_NICEST );

    // Use GL_NEAREST to see the voxels
    glEnable( GL_TEXTURE_3D ); // Enable Texture Mapping
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // Sets the wrap parameter for texture coordinate
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE  );
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE  );
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE  );

    glEnable( GL_POLYGON_SMOOTH_HINT );
    glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}



void Volumn::render_volumn( const float& dx,
                            const float& dy,
                            const float& dz )
{

    glBindTexture(GL_TEXTURE_3D, texture);
    glBegin(GL_QUADS);
    for( float i=0.0f; i<=sx-1; i+=dx )
    {
        glTexCoord3f( 1.0f*(i+0.5f)/(float)texture_sx,
                      0.0f,
                      0.0f );
        glVertex3f( i, 0.0f, 0.0f );

        glTexCoord3f( 1.0f*(i+0.5f)/(float)texture_sx,
                      1.0f*((float)sy-0.5f)/(float)texture_sy,
                      0.0f );
        glVertex3f( i, 1.0f*((float)sy-1), 0.0f );

        glTexCoord3f( 1.0f*(i+0.5f)/(float)texture_sx,
                      1.0f*((float)sy-0.5f)/(float)texture_sy,
                      1.0f*((float)sz-0.5f)/(float)texture_sz );
        glVertex3f( i, 1.0f*((float)sy-1), 1.0f*((float)sz-1) );

        glTexCoord3f( 1.0f*(i+0.5f)/(float)texture_sx, 0.0f,
                      1.0f*((float)sz-0.5f)/(float)texture_sz );
        glVertex3f( i, 0.0f, 1.0f*((float)sz-1) );
    }
    for( float i=0.0f; i<=sy-1; i+=dy )
    {
        glTexCoord3f( 0.0f,
                      1.0f*(i+0.5f)/(float)texture_sy,
                      0.0f );
        glVertex3f( 0.0f, i, 0.0f );

        glTexCoord3f( 1.0f*((float)sx-0.5f)/(float)texture_sx,
                      1.0f*(i+0.5f)/(float)texture_sy,
                      0.0f );
        glVertex3f( 1.0f*((float)sx-1), i, 0.0f );

        glTexCoord3f( 1.0f*((float)sx-0.5f)/(float)texture_sx,
                      1.0f*(i+0.5f)/(float)texture_sy,
                      1.0f*((float)sz-0.5f)/(float)texture_sz );
        glVertex3f( 1.0f*((float)sx-1), i, 1.0f*((float)sz-1) );

        glTexCoord3f( 0.0f,
                      1.0f*(i+0.5f)/(float)texture_sy,
                      1.0f*((float)sz-0.5f)/(float)texture_sz );
        glVertex3f( 0.0f,        i, 1.0f*((float)sz-1) );
    }
    for( float i=0.0f; i<=sz-1; i+=dz )
    {
        glTexCoord3f( 0.0f,
                      0.0f,
                      1.0f*(i+0.5f)/(float)texture_sz );
        glVertex3f( 0.0f, 0.0f, i );

        glTexCoord3f( 1.0f*((float)sx-0.5f)/(float)texture_sx,
                      0.0f,
                      1.0f*(i+0.5f)/(float)texture_sz );
        glVertex3f( 1.0f*((float)sx-1), 0.0f, i );

        glTexCoord3f( 1.0f*((float)sx-0.5f)/(float)texture_sx,
                      1.0f*((float)sy-0.5f)/(float)texture_sy,
                      1.0f*(i+0.5f)/(float)texture_sz );
        glVertex3f( 1.0f*((float)sx-1),
                    1.0f*((float)sy-1),
                    i );

        glTexCoord3f( 0.0f,
                      1.0f*((float)sy-0.5f)/(float)texture_sy,
                      1.0f*(i+0.5f)/(float)texture_sz );
        glVertex3f( 0.0f, 1.0f*((float)sy-1), i );
    }
    glEnd();
    glBindTexture( GL_TEXTURE_3D, 0 );


}

void Volumn::render_outline(void)
{
    const float x_min = -0.5f;
    const float y_min = -0.5f;
    const float z_min = -0.5f;
    const float X_MAX = (float)sx - 1 - x_min;
    const float Y_MAX = (float)sy - 1 - y_min;
    const float Z_MAX = (float)sz - 1 - z_min;
    // left borders
    glBegin(GL_LINE_LOOP);
    glVertex3f( x_min, y_min, z_min );
    glVertex3f( x_min, Y_MAX, z_min );
    glVertex3f( x_min, Y_MAX, Z_MAX );
    glVertex3f( x_min, y_min, Z_MAX );
    glEnd();
    // right borders
    glBegin(GL_LINE_LOOP);
    glVertex3f( X_MAX, y_min, z_min );
    glVertex3f( X_MAX, Y_MAX, z_min );
    glVertex3f( X_MAX, Y_MAX, Z_MAX );
    glVertex3f( X_MAX, y_min, Z_MAX );
    glEnd();
    // parrenl lines to x-axix
    glBegin(GL_LINES);
    glVertex3f( x_min, y_min, z_min );
    glVertex3f( X_MAX, y_min, z_min );
    glVertex3f( x_min, Y_MAX, z_min );
    glVertex3f( X_MAX, Y_MAX, z_min );
    glVertex3f( x_min, Y_MAX, Z_MAX );
    glVertex3f( X_MAX, Y_MAX, Z_MAX );
    glVertex3f( x_min, y_min, Z_MAX );
    glVertex3f( X_MAX, y_min, Z_MAX );
    glEnd();
}


void Volumn::render(void)
{
    glPushMatrix();
    glScalef( scale, scale, scale );


    if( render_mode == MIP )
    {
        /* Visualizing the data with Maximum Intensity Projection (MIP). */
        glColor3f( 1.0f, 1.0f, 1.0f );

        float dx = this->size_x() / 150;
        float dy = this->size_y() / 150;
        float dz = this->size_z() / 150;

        /* The number of slices is based on scale. A finner detail is shown
           while we zoom in and vice versa. */
        dx /= this->ptrCam->scale;
        dy /= this->ptrCam->scale;
        dz /= this->ptrCam->scale;

        /* Show more number of slices toward the direction where the
           camera is looking at. */
        const cv::Vec3f vx( this->ptrCam->vec_x );
        const cv::Vec3f vy( this->ptrCam->vec_y );
        const cv::Vec3f vz = vx.cross( vy );
        const float sx = std::abs( vz[0] );
        const float sy = std::abs( vz[1] );
        const float sz = std::abs( vz[2] );
        dx = dx * sx + dx * 4 * ( 1- sx );
        dy = dy * sy + dy * 4 * ( 1- sy );
        dz = dz * sz + dz * 4 * ( 1- sz );

        render_volumn( dx, dy, dz );

        // draw a frame of the volumn
        glColor3f( 0.2f, 0.2f, 0.2f );
        render_outline();
    }
    else if( render_mode == CrossSection )
    {
        // Yuchen: This rendering Mode requires the information of camera
        if( ptrCam==NULL ) return;

        // retrive camera infomation
        cv::Vec3f center, vz;
        center[0] = ptrCam->t[0];
        center[1] = ptrCam->t[1];
        center[2] = ptrCam->t[2];
        vz[0] = ptrCam->vec_x[1]*ptrCam->vec_y[2]
                - ptrCam->vec_x[2]*ptrCam->vec_y[1];
        vz[1] = ptrCam->vec_x[2]*ptrCam->vec_y[0]
                - ptrCam->vec_x[0]*ptrCam->vec_y[2];
        vz[2] = ptrCam->vec_x[0]*ptrCam->vec_y[1]
                - ptrCam->vec_x[1]*ptrCam->vec_y[0];
        /* Get the cross section of cube. It can be a point, a line, a triangle,
           a rectangle, a pentagon, a hexagon and etc. */
        std::vector<cv::Vec3f> points = intersectPoints( center, vz );

        glEnable(GL_DEPTH_TEST);

        // draw the cross section
        glColor3f( 1.0f, 1.0f, 1.0f );
        glBindTexture(GL_TEXTURE_3D, texture);
        glBegin( GL_TRIANGLE_FAN );
        for( unsigned int i=0; i<points.size(); i++ )
        {
            glTexCoord3f( points[i][0] / (float)texture_sx,
                          points[i][1] / (float)texture_sy,
                          points[i][2] / (float)texture_sz );
            glVertex3f( points[i][0], points[i][1], points[i][2] );
        }
        glEnd();
        glBindTexture( GL_TEXTURE_3D, 0 );

        // draw the frame of the box
        glColor3f( 0.0f, 0.0f, 0.8f );
        render_outline();

        /* We want to boarder to be visible all the time; therefore,
           depth_test is disabled. */
        glDisable(GL_DEPTH_TEST);
        // draw the boundary of the cross section
        glColor3f( 0.3f, 0.3f, 0.3f );
        glBegin( GL_LINE_LOOP );
        for( unsigned int i=0; i<points.size(); i++ )
        {
            glVertex3f( points[i][0], points[i][1], points[i][2] );
        }
        glEnd();

    }
    else if ( render_mode==Surface )
    {
        glColor3f( 1.0f, 1.0f, 1.0f );
        render_volumn( (float)sx-1.0f,
                       (float)sy-1.0f,
                       (float)sz-1.0f );
        glColor3f( 0.2f, 0.2f, 0.2f );
        render_outline();
    }

    glPopMatrix();
}


bool Volumn::update_data( unsigned char* im_data )
{
    if( !data ) return false;;

    // Update texture data
    for( int z=0; z<sz; z++ )
    {
        for( int y=0; y<sy; y++ )
        {
            memcpy( data + z*texture_sy*texture_sx + y*texture_sx,
                    im_data + z*sy*sx + y*sx,
                    sx );
        }
    }

    // update texture for rendering
    glBindTexture(GL_TEXTURE_3D, texture);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE,
                 texture_sx, texture_sy, texture_sz, 0,
                 GL_LUMINANCE, GL_UNSIGNED_BYTE, data );
    return true;
}

}
