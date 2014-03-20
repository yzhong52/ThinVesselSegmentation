#pragma once
#include <string>

namespace GLViewer
{
	///////////////////////////////////////////////////
	// Saving OpenGL Frame Buffer to Video 
	///////////////////////////////////////////////////
	class VideoSaver {
		// frame rate
		static int const fps = 20;
		// name of the video
		std::string video_file_name;
		// currently saving frame
		int current_frame;
		// total number of frames to be saved
		int total_frames;

		// initialized in void init(int, int)
		int width;
		int height;

		enum State{
			Rendering,
			isAboutToStop, 
			isStopped
		} state; 

		bool autoRotate; 
	public:
		VideoSaver( ) : state( isStopped ), autoRotate( false ) { }

		unsigned int size_x(void) const { return 0; }
		unsigned int size_y(void) const { return 0; }
		unsigned int size_z(void) const { return 0; }

		// initialize video saver
		void init(int w, int h, std::string filename, int maxNumFrames );
		inline void stop( void ) { state = isAboutToStop; }
		inline bool isDone( void ) const { return state==isStopped; } 
		inline bool isRendering( void ) const { return (state==Rendering); } 

		// saving video
		void saveBuffer(void);

		// take screen shot
		void takeScreenShot( int w, int h); 

		inline const bool& isAutoRotate( void ) const { return autoRotate; } 
		inline const void setAutoRotate( bool flag ) { autoRotate = flag; } 
	};
}

