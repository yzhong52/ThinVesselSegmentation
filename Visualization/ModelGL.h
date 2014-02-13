///////////////////////////////////////////////////////////////////////////////
// ModelGL.h
// =========
// Model component of OpenGL
// 
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-07-10
// UPDATED: 2013-11-27
///////////////////////////////////////////////////////////////////////////////

#ifndef MODEL_GL_H
#define MODEL_GL_H

class ModelGL
{
public:
    ModelGL();
    ~ModelGL();

    void init();                                    // initialize OpenGL states
    void setViewport();
    void resizeWindow(int width, int height);
    void draw();

	// Mouse Control Message
	int mouse_pos_x, mouse_pos_y; 
	int mouse_down_x, mouse_down_y; 
	// Mouse Up
	void mouseUp_RightButton( int x, int y );
	void mouseUp_LeftButton( int x, int y );
	void mouseUp_MiddleButton( int x, int y );
	// Mouse Down
	void mouseDown_RightButton( int x, int y );
	void mouseDown_LeftButton( int x, int y );
	void mouseDown_MiddleButton( int x, int y );
	// Mouse Move
	void mouseMove_LeftButton( int x, int y );
	void mouseMove_RightButton( int x, int y );
	void mouseMove_MiddleButton( int x, int y );
	// Mouse Wheel 
	void mouseWheel_Up( void ); 
	void mouseWheel_Down( void ); 

	short windowCenterMin; 
	short windowCenterMax; 
	void updateWindowCenterMin( int position );
	void updateWindowCenterMax( int position );
	void updateWindowCenter( int min, int max );
	void windowCenterUpdateData();

protected:

private:
    // members
    int windowWidth;
    int windowHeight;
    bool windowResized;
    
};
#endif
