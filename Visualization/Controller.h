///////////////////////////////////////////////////////////////////////////////
// Controller.h
// ============
// a base class of Windows Controller component
// This class is a bridge between window procedure and the application.
// Controller interacts with the events of the application. It receives the
// input from the user and translates it to Model and View in order to perform
// actions based on that input. Controller informs Model with a setter
// function in order to update its associated value. And, it also notifies
// View to update visuals.
//
// This class does not handle any message here, therefore the client must
// create a derived class and handle the specific messages.
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2005-03-31
// UPDATED: 2013-03-15
///////////////////////////////////////////////////////////////////////////////

#ifndef WIN_CONTROLLER_H
#define WIN_CONTROLLER_H

#include <windows.h>


namespace Win
{
    class Controller
    {
    public:
        // ctor/dtor
        Controller();
        virtual ~Controller ();

        void setHandle(HWND handle);                            // set window handle

        virtual int close();                                    // for WM_CLOSE
        virtual int command(int id, int cmd, LPARAM msg);       // for WM_COMMAND: id, cmd, msg
        virtual int contextMenu(HWND handle, int x, int y);     // for WM_CONTEXTMENU: handle, x, y
        virtual int create();                                   // for WM_CREATE
        virtual int destroy();                                  // for WM_DESTROY
        virtual int enable(bool flag);                          // for WM_ENABLE: flag(true/false)
        virtual int eraseBkgnd(HDC hdc);                        // for WM_ERASEBKGND: HDC
        virtual int hScroll(WPARAM wParam, LPARAM lParam);      // for WM_HSCROLL
        virtual int keyDown(int key, LPARAM lParam);            // for WM_KEYDOWN: keyCode, detailInfo
        virtual int keyUp(int key, LPARAM lParam);              // for WM_KEYUP: keyCode, detailInfo
        virtual int lButtonDown(WPARAM state, int x, int y);    // for WM_LBUTTONDOWN: state, x, y
        virtual int lButtonUp(WPARAM state, int x, int y);      // for WM_LBUTTONUP: state, x, y
        virtual int mButtonDown(WPARAM state, int x, int y);    // for WM_MBUTTONDOWN: state, x, y
        virtual int mButtonUp(WPARAM state, int x, int y);      // for WM_MBUTTONUP: state, x, y
        virtual int mouseMove(WPARAM state, int x, int y);      // for WM_MOUSEMOVE: state, x, y
        virtual int mouseWheel(int state, int d, int x, int y); // for WM_MOUSEWHEEL: state, delta, x, y
        virtual int notify(int id, LPARAM lParam);              // for WM_NOTIFY: controllerID, NMHDR
        virtual int paint();                                    // for WM_PAINT
        virtual int rButtonDown(WPARAM wParam, int x, int y);   // for WM_RBUTTONDOWN: state, x, y
        virtual int rButtonUp(WPARAM wParam, int x, int y);     // for WM_RBUTTONUP: state, x, y
        virtual int size(int w, int h, WPARAM wParam);          // for WM_SIZE: width, height, type(SIZE_MAXIMIZED...)
        virtual int timer(WPARAM id, LPARAM lParam);            // for WM_TIMER: ID, ptr to callback ftn
        virtual int vScroll(WPARAM wParam, LPARAM lParam);      // for WM_VSCROLL


    protected:
        HWND handle;                                            // window handle to map window to controller

    private:
        // Controller class must reference to Model and View components.
        // Controller receives an event from the user and translates it to
        // Model and View to perform actions based on that input.
        //Model model;
        //View  view;
    };

    ///////////////////////////////////////////////////////////////////////////
    // inline functions
    ///////////////////////////////////////////////////////////////////////////
    inline void Controller::setHandle(HWND hwnd) { handle = hwnd; }
    inline int Controller::close() { ::DestroyWindow(handle); return 0; }
    inline int Controller::command(int id, int cmd, LPARAM msg) { return 0; }
    inline int Controller::contextMenu(HWND handle, int x, int y) { return 0; }
    inline int Controller::create() { return 0; }
    inline int Controller::destroy() { return 0; }
    inline int Controller::enable(bool flag) { return 0; }
    inline int Controller::eraseBkgnd(HDC hdc) { return 0; }
    inline int Controller::hScroll(WPARAM wParam, LPARAM lParam) { return 0; }
    inline int Controller::keyDown(int key, LPARAM lParam) { return 0; }
    inline int Controller::keyUp(int key, LPARAM lParam) { return 0; }
    inline int Controller::lButtonDown(WPARAM wParam, int x, int y) { return 0; }
    inline int Controller::lButtonUp(WPARAM wParam, int x, int y) { return 0; }
    inline int Controller::mButtonDown(WPARAM wParam, int x, int y) { return 0; }
    inline int Controller::mButtonUp(WPARAM wParam, int x, int y) { return 0; }
    inline int Controller::mouseMove(WPARAM keyState, int x, int y) { return 0; }
    inline int Controller::mouseWheel(int state, int delta, int x, int y) { return 0; }
    inline int Controller::notify(int id, LPARAM lParam) { return 0; }
    inline int Controller::paint() { return 0; }
    inline int Controller::rButtonDown(WPARAM wParam, int x, int y) { return 0; }
    inline int Controller::rButtonUp(WPARAM wParam, int x, int y) { return 0; }
    inline int Controller::size(int w, int h, WPARAM type) { return 0; }
    inline int Controller::timer(WPARAM id, LPARAM lParam) { return 0; }
    inline int Controller::vScroll(WPARAM wParam, LPARAM lParam) { return 0; }
}
#endif
