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
    void setCamera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ);
    void setViewport(int width, int height);
    void resizeWindow(int width, int height);
    void draw();

    void setMouseLeft(bool flag) { mouseLeftDown = flag; };
    void setMouseRight(bool flag) { mouseRightDown = flag; };
    void setMousePosition(int x, int y) { mouseX = x; mouseY = y; };
    void setDrawMode(int mode);
    void animate(bool flag) { animateFlag = flag; };

    void rotateCamera(int x, int y);
    void zoomCamera(int dist);

    void setBackgroundRed(float value);             // change background colour
    void setBackgroundGreen(float value);
    void setBackgroundBlue(float value);

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

protected:

private:
    // member functions
    //void initLights();                              // add a white light ti scene
    //unsigned int createEarthDL();
    //unsigned int loadTextureBmp(const char *filename);

    // members
    int windowWidth;
    int windowHeight;
    bool animateFlag;                               // on/off animation
    bool changeDrawMode;
    int drawMode;
    unsigned int listId;                            // display list ID
    bool mouseLeftDown;
    bool mouseRightDown;
    int mouseX;
    int mouseY;
    float cameraAngleX;
    float cameraAngleY;
    float cameraDistance;
    float bgColor[4];
    bool bgFlag;
    bool windowResized;
    unsigned char* frameBuffer;                     // framebuffer to store RGBA color
    int bufferSize;                                 // framebuffer size in bytes
};
#endif
