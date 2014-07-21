#include "GLViewer.h"
#include "GLCamera.h"

#include <iostream>
#include <sstream>
using namespace std;

#include <time.h>
#include <GL/glut.h> // Glut Library
#include "GLVideoSaver.h"

namespace GLViewer
{

// an array of objects that need to be render
vector<Object*> obj;

// maximum number of viewports supported
const int maxNumViewports = 9;

// maximum number of objects suported
const int maxNumObjects = 9;

// number of viewports that is currently available
int numViewports = 1;

// isDisplayObject[i][j] indicates that whether or not display object i in viewport j
bool isDisplayObj[maxNumViewports][maxNumObjects];

// Size of the 3D data
unsigned sx = 0;
unsigned sy = 0;
unsigned sz = 0;

/////////////////////////////////////////
// Camera Controls by Mouse
///////////////////////
GLCamera camera;
int mouse_pos_x = 0;
int mouse_pos_y = 0;

/////////////////////////////////////////
// Initial Window Size
///////////////////////
int width = 1280;
int height = 720;

VideoSaver* videoSaver = NULL;

// Whether or not display the axis in the scene
bool isAxis = false;

// Whether or not to take a screen shot from the current rendering result
bool isSaveFrame = false;

void render(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear The Screen And The Depth Buffer

    camera.push_matrix();

    // rending objects
    if( numViewports == 4 )
    {
        // if there are 4 viewports, display the objects in two rows
        // each row with 2 columns
        for( unsigned int i=0; i<2; i++ )
        {
            for( unsigned int j=0; j<2; j++ )
            {
                glViewport (width/2*i, height/2*j, width/2, height/2);

                unsigned int vp_index = i + 2*(1-j);

                for( unsigned int obj_index=0; obj_index<obj.size(); obj_index++ )
                {
                    if( isDisplayObj[vp_index][obj_index] ) obj[obj_index]->render();
                }

                // draw rotation axis
                if( isAxis ) camera.draw_axis();
            }
        }
    }
    else
    {
        // if there are 1, 2, or 3 view ports, display then in a row
        for( int i=0; i<numViewports; i++ )
        {
            // For view port i
            glViewport (i*width/numViewports, 0, width/numViewports, height);
            for( unsigned int j=0; j<obj.size(); j++ )
            {
                if( isDisplayObj[i][j] ) obj[j]->render();
            }

            // draw rotation axis
            if( isAxis ) camera.draw_axis();
        }
    }
    camera.pop_matrix();
    camera.rotate_scene();

    // saving frame buffer as video
    if( videoSaver )
    {
        videoSaver->saveBuffer();
    }

    // take a screen shot
    if( isSaveFrame && videoSaver )
    {
        videoSaver->takeScreenShot( width, height );
        isSaveFrame = false;
    }

    glutSwapBuffers();
}

void startCaptureVideo( int maxNumFrames )
{
    static int index = 0;

    stringstream videoName;
    videoName << "output/video" << ++index << ".avi";

    cout << "Begin to create video '" << videoName.str() << "'" << endl;

    if( !videoSaver ) videoSaver = new VideoSaver();
    videoSaver->init(width, height, videoName.str(), maxNumFrames );
}

void mouse_click(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON)   // mouse left button
    {
        static int mouse_down_x;
        static int mouse_down_y;
        if(state == GLUT_DOWN)
        {
            // The camera is already rotating around the center of the scence,
            // The original rotation speed is zero. Rotatio speed is updated with
            // mouse click (left button).
            camera.setNavigationMode( GLCamera::Rotate );
            mouse_pos_x = x;
            mouse_pos_y = y;
            mouse_down_x = x;
            mouse_down_y = y;
        }
        else if( state == GLUT_UP )
        {
            // stop tracking mouse move for rotation
            camera.setNavigationMode( GLCamera::None );

            // Double click to stop camera rotation
            if( mouse_down_x==x && mouse_down_y==y )
            {
                // if the user click and release the mouse at the same point
                camera.setRotation( 0, 0 ); // stop rotation
            }
        }
    }
    else if(button == GLUT_RIGHT_BUTTON)     // mouse right button
    {
        // right click to navigate the scene (moving up, down, left or right)
        if( state == GLUT_DOWN )
        {
            camera.setNavigationMode( GLCamera::MoveAside );
            mouse_pos_x = x;
            mouse_pos_y = y;
        }
        else
        {
            camera.setNavigationMode( GLCamera::None );
        }
    }
    else if( button==GLUT_MIDDLE_BUTTON )     // center button
    {
        // click with middle buttong to navigate the scene (forward or backward)
        if( state == GLUT_DOWN )
        {
            camera.setNavigationMode( GLCamera::MoveForward );
            mouse_pos_x = x;
            mouse_pos_y = y;
        }
        else
        {
            camera.setNavigationMode( GLCamera::None );
        }
    }
    else if ( button==3 )
    {
        // mouse wheel scrolling up to zoom in
        camera.zoomIn();
    }
    else if ( button==4 )
    {
        // mouse wheel scrooling down to zoom out
        camera.zoomOut();
    }
}

// the mouse_move function will only be called when at least one button of the
// mouse is holded down
void mouse_move(int x, int y)
{
    if( camera.getNavigationMode() == GLCamera::Rotate )
    {
        camera.setRotation( 1.0f*(x - mouse_pos_x), 1.0f*(y - mouse_pos_y) );
        glutPostRedisplay();
    }
    else if( camera.getNavigationMode() == GLCamera::MoveAside )
    {
        camera.translate_aside( x - mouse_pos_x, y - mouse_pos_y );
    }
    else if( camera.getNavigationMode()==GLCamera::MoveForward )
    {
        camera.translate_forward( x - mouse_pos_x, y - mouse_pos_y );
    }

    // update mouse position
    mouse_pos_x = x;
    mouse_pos_y = y;
}


void reset_projection(void)
{
    // Select The Projection Transformation
    glMatrixMode(GL_PROJECTION);
    // Reset The Projection Matrix
    glLoadIdentity();

    GLfloat maxVal = max( sx, max(sy, sz) ) * 0.8f;

    GLfloat ratio = 1;
    if ( numViewports==4 )
    {
        ratio = (GLfloat)width / (GLfloat)height;
    }
    else
    {
        ratio = (GLfloat)width / (GLfloat)height / numViewports;
    }

    glOrtho( -maxVal*ratio, maxVal*ratio, -maxVal, maxVal, -maxVal, maxVal);

    glMatrixMode(GL_MODELVIEW);     // Select The Model Transformation
    glutPostRedisplay();
}

void reset_modelview(void)
{
    camera.resetModelview( (GLfloat)sx, (GLfloat)sy, (GLfloat)sz );
    glutPostRedisplay();
}

void reshape(int w, int h)
{
    // limit the minimum size of the windows
    width = max(50, w);
    height = max(50, h);

    // Reset Projection Transformation
    reset_projection();
}

void keyboard(unsigned char key, int x, int y)
{
    // use number key 1~9 to control which object to display
    if( key >= '1' && key <= '9' )
    {
        // object index
        unsigned int index = key - '1';

        if( index >= obj.size() )  return;

        if (glutGetModifiers() == GLUT_ACTIVE_ALT )
        {
            obj[index]->keyboard( '\t' );
        }
        else
        {
            isDisplayObj[numViewports-1][index] = !isDisplayObj[numViewports-1][index];
            cout << ((isDisplayObj[numViewports-1][index]) ? "Display " : "Hide ");
            cout << "object " << index << " in view port " << numViewports-1 << endl;
        }
    }

    switch (key)
    {
    case ' ':
        reset_projection();
        reset_modelview();
        break;
    case 'a':
        // toggle on/off the axis
        isAxis = !isAxis;
        break;
    case '\t':
        // increment the number of viewports: 1, 2, 3...
        numViewports = numViewports % maxNumViewports + 1;
        reset_projection();
        break;
    case 's':
    case 'S':
        isSaveFrame = true;
        break;

    /////////////////////////////
    // saving video
    /////////////////////////////
    case 'v':
    case 'V':
        if( !videoSaver || videoSaver->isDone() )
        {
            startCaptureVideo( );
        }
        else if( videoSaver->isRendering() )
        {
            cout << "Video render complete. " << endl;
            videoSaver->stop();
        }
        break;

    case 27:
        cout << "Rednering done. Thanks you for using GLViewer. " << endl;
        exit(0);
        break;
    }
}

void dispay( vector<Object*> objects, int w, int h )
{
    camera.setNavigationMode( GLCamera::Rotate );

    width = w;
    height = h;

    if( objects.size()==0 )
    {
        std::cerr << "No Objects For Rendering... " << std::endl;
        return;
    }
    else if( objects.size()> maxNumObjects )
    {
        std::cerr << "Too many objects to visulized... " << std::endl;
        return;
    }


    obj = objects;

    std::fill( isDisplayObj[0],
               isDisplayObj[0]+sizeof(isDisplayObj)*sizeof(isDisplayObj[0][0]),
               false );
    for( unsigned int i=0; i<maxNumViewports; i++ )
    {
        // put the i-th object in the i-th viewport
        if( i < objects.size() ) isDisplayObj[i][i] = true;
        else isDisplayObj[i][0] = true;
    }

    ///////////////////////////////////////////////
    // glut Initializaitons
    ///////////////
    int argc = 1;
    char* argv[1] = { NULL };
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( argv[0] );
    glewInit();
    // Register Recall Funtions
    glutReshapeFunc( reshape );
    // have post_draw_func, which is for saving videos
    glutKeyboardFunc( keyboard );
    // register mouse fucntions
    glutMouseFunc( mouse_click );
    glutMotionFunc( mouse_move );
    // render func
    glutIdleFunc( render );
    glutDisplayFunc( render );


    // Global Inits - Antialiasing
    glEnable( GL_LINE_SMOOTH );
    glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
    // The order of the following settings do matters
    // Setting up the size of the scence
    for( unsigned int i=0; i<obj.size(); i++ )
    {
        obj[i]->init(); // additionol init settings by objects
        sx = max( sx, obj[i]->size_x() );
        sy = max( sy, obj[i]->size_y() );
        sz = max( sz, obj[i]->size_z() );
    }

    // reset the modelview and projection
    reset_projection();
    reset_modelview();

    // Initial Rotation (Do as you want ); Now it is faciton the x-y plane
    camera.rotate_x(-90); // rotate around x-axis for 90 degrees

    cout << "Redenring Begin..." << endl;
    cout << "======================= Instructions =======================" << endl;
    cout << "   Mouse Controls: " << endl;
    cout << "       LEFT Button       - Rotation " << endl;
    cout << "       RIGHT Button      - Translation (aside) " << endl;
    cout << "       Middle Button     - Translation (forward/backward) " << endl;
    cout << "   Keyboard Controls: " << endl;
    //cout << "       TAB               - Toggle on/off The Second Viewport" << endl;
    cout << "       a                 - Toggle on/off Rotation Center " << endl;
    cout << "       SPACE             - Reset Projection Matrix " << endl;
    cout << "       1,2,3...          - Toggle on/off objects " << endl;
    cout << "       ALT + 1,2,3...  - Change Rendering Mode for a object " << endl;
    cout << "       v/V               - toggle start/stop saving video" << endl;
    cout << "       ESC               - Exit " << endl;

    // Enter glut main loop: No Code Will Be Executed After This Line!
    glutMainLoop();
}

}



