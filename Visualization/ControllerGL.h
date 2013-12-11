///////////////////////////////////////////////////////////////////////////////
// ControllerGL.h
// ==============
// Derived Controller class for OpenGL window
// It is the controller of OpenGL rendering window. It initializes DC and RC,
// when WM_CREATE called, then, start new thread for OpenGL rendering loop.
//
// When this class is constructed, it gets the pointers to model and view
// components.
//
//  AUTHOR: Song Ho Ahn (song.ahn@gamil.com)
// CREATED: 2006-07-09
// UPDATED: 2013-11-28
///////////////////////////////////////////////////////////////////////////////

#ifndef WIN_CONTROLLER_GL_H
#define WIN_CONTROLLER_GL_H

#include "Controller.h"
#include "ViewGL.h"
#include "ModelGL.h"


namespace Win
{
    class ControllerGL : public Controller
    {
    public:
        ControllerGL(ModelGL* model, ViewGL* view);
        ~ControllerGL() {};

        int close();                                // close the RC and OpenGL window
        int command(int id, int cmd, LPARAM msg);   // for WM_COMMAND
        int create();                               // create RC for OpenGL window and start new thread for rendering
        int paint();
        int size(int w, int h, WPARAM wParam);
        int lButtonDown(WPARAM state, int x, int y);
        int lButtonUp(WPARAM state, int x, int y);
        int rButtonDown(WPARAM state, int x, int y);
        int rButtonUp(WPARAM state, int x, int y);
        int mouseMove(WPARAM state, int x, int y);

        // not needed anymore
        //void setModel(ModelGL* model) { if(model) modelGL = model; };   // get reference of ModelGL
        //void setView(ViewGL* view) { if(view) viewGL = view; };         // get reference of ViewGL

    private:
        static void threadFunction(void* arg);      // static thread function, it will route to member function, runThread()
        void runThread();                           // thread for OpenGL rendering

        ModelGL* model;                             //
        ViewGL* view;                               //
        HANDLE threadHandle;
        unsigned int threadId;
        volatile bool loopFlag;                     // rendering loop flag
    };
}

#endif
