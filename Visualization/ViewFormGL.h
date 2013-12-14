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
        // void updateTrackbars(HWND handle, int position);

		void updateTrackbarWindowCenterMin( int position );
		void updateTrackbarWindowCenterMax( int position );

		inline int getTrackbarWindowCenterMin(void) {
			return trackbarWindowCenterMin.getPos(); 
		}
		inline int getTrackbarWindowCenterMax(void) {
			return trackbarWindowCenterMax.getPos(); 
		}

    protected:

    private:
        // controls        
		// Yuchen: Tracker For Window Center
		Win::Trackbar trackbarWindowCenterMin; 
		Win::Trackbar trackbarWindowCenterMax; 
    };
}

#endif
