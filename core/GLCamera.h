#pragma once

/////////////////////////////////////
// Glew Library
#include <GL/glew.h> // For Texture 3D and Blending_Ext

/////////////////////////////////////
// OpenGL Library
#if _MSC_VER && !__INTEL_COMPILER
#pragma comment(lib, "glew32.lib")
#include <windows.h>		// Header File For Windows
#endif

#include <GL/gl.h>			// Header File For The OpenGL Library
#include <GL/glu.h>			// Header File For The GLu Library


class GLCamera
{
public:
    GLCamera(const float& rotation_speed = 0.01f);
    ~GLCamera(void);

    // Navigation Mode
    enum NavigationMode
    {
        None,
        MoveAside,
        MoveForward,
        Rotate
    } navigationMode;

    // setter and getters
    inline void setNavigationMode( NavigationMode nMode );
    inline NavigationMode getNavigationMode(void);

    inline void push_matrix()
    {
        glScalef( scale, scale, scale );
        glTranslatef( -t[0], -t[1], -t[2] );
    }

    inline void pop_matrix()
    {
        glTranslatef( t[0], t[1], t[2] );
        glScalef( 1.0f/scale, 1.0f/scale, 1.0f/scale );
    }

    void zoomIn(void);
    void zoomOut(void);

    void rotate_scene(void);

    inline void setRotation( GLfloat rotation_x, GLfloat rotation_y )
    {
        rot_x = rotation_x * rot_speed;
        rot_y = rotation_y * rot_speed;
    }

    inline void translate_aside( int translate_x, int translate_y )
    {
        GLfloat tx = -translate_x * translate_speed;
        GLfloat ty =  translate_y * translate_speed;
        // update the position of the center
        t[0] += tx * vec_x[0];
        t[1] += tx * vec_x[1];
        t[2] += tx * vec_x[2];
        t[0] += ty * vec_y[0];
        t[1] += ty * vec_y[1];
        t[2] += ty * vec_y[2];
    }
    inline void translate_forward( int translate_x, int translate_y )
    {
        GLfloat tx = translate_x * translate_speed;
        GLfloat ty = translate_y * translate_speed;
        GLfloat vec_z[3];
        vec_z[0] = vec_x[1]*vec_y[2] - vec_x[2]*vec_y[1];
        vec_z[1] = vec_x[2]*vec_y[0] - vec_x[0]*vec_y[2];
        vec_z[2] = vec_x[0]*vec_y[1] - vec_x[1]*vec_y[0];
        // update translation vector
        t[0] += (tx+ty) * vec_z[0];
        t[1] += (tx+ty) * vec_z[1];
        t[2] += (tx+ty) * vec_z[2];
    }

    inline void resetModelview( GLfloat cx, GLfloat cy, GLfloat cz )
    {
        t[0] = 0.5f * cx;
        t[1] = 0.5f * cy;
        t[2] = 0.5f * cz;

        // rotation axis
        vec_y[0] = 0;
        vec_y[1] = 1;
        vec_y[2] = 0;
        vec_x[0] = 1;
        vec_x[1] = 0;
        vec_x[2] = 0;
        // rotation parameters
        rot_x = 0;
        rot_y = 0;

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity(); // clear the identity matrix.
        gluLookAt( 0, 0, 1, /*eye position*/
                   0, 0, 0, /*Center of the object*/
                   0, 1, 0 ); /*Up Vector*/
    }


    inline void draw_axis( void )
    {
        glTranslatef( t[0], t[1], t[2] );
        // Draw Rotation Center with two axis
        glBegin(GL_LINES);
        glColor3f( 1.0, 0.0, 0.0 );
        glVertex3i( 0, 0, 0 );
        glVertex3f( vec_x[0]*10, vec_x[1]*10, vec_x[2]*10 );
        glColor3f( 0.0, 1.0, 0.0 );
        glVertex3i( 0, 0, 0 );
        glVertex3f( vec_y[0]*10, vec_y[1]*10, vec_y[2]*10 );
        glEnd();
        glTranslatef( -t[0], -t[1], -t[2] );
    }

    void rotate_y( float degree );
    void rotate_x( float degree );

private:
    // Rotation
    GLfloat	rot_x;
    GLfloat	rot_y;
    GLfloat rot_speed;

public:
    // Rotation Axis
    GLfloat vec_y[3];
    GLfloat vec_x[3];
    // Translation
    GLfloat t[3];
    GLfloat translate_speed;

    int elapsedTick;
    float scale;
};



// setter and getters
inline void GLCamera::setNavigationMode( NavigationMode nMode )
{
    navigationMode = nMode;
}
inline GLCamera::NavigationMode GLCamera::getNavigationMode(void)
{
    return navigationMode;
}


