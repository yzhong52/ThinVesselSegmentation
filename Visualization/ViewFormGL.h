///////////////////////////////////////////////////////////////////////////////
// ViewFormGL.h
// ============
// View component of OpenGL dialog window
//
//  AUTHORL Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-07-10
// UPDATED: 2006-08-15
///////////////////////////////////////////////////////////////////////////////

#ifndef VIEW_FORM_GL_H
#define VIEW_FORM_GL_H

#include <windows.h>
#include "Controls.h"

namespace Win
{
    class ViewFormGL
    {
    public:
        ViewFormGL();
        ~ViewFormGL();

        void initControls(HWND handle);         // init all controls

        // update controls on the form
        void animate(bool flag);
        void updateTrackbars(HWND handle, int position);

    protected:

    private:
        // controls
        Win::Button buttonAnimate;
        Win::RadioButton radioFill;
        Win::RadioButton radioWireframe;
        Win::RadioButton radioPoint;
        Win::Trackbar trackbarRed;
        Win::Trackbar trackbarGreen;
        Win::Trackbar trackbarBlue;
    };
}

#endif
