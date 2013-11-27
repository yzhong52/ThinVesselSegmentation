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
		std::string video_name;
		// dufation of saving
		double duration;
		// currently saving frame
		int current_frame;
		// total number of frames to be saved
		int total_frames;
		// initialized in void init(int, int)
		bool isInit;
		int width;
		int height;
		// auto rotaion
		bool autoRotate;
	public:
		VideoSaver( std::string video_name, double duration = 18, bool autoRotate = true ) 
			: isInit( false )
			, video_name( video_name )
			, duration( duration)
			, autoRotate( autoRotate )
		{
			total_frames = int( fps * duration ); 
		}

		unsigned int size_x(void) const { return 0; }
		unsigned int size_y(void) const { return 0; }
		unsigned int size_z(void) const { return 0; }

		void init(int weight, int height);
		void saveBuffer(void);
	};
}

