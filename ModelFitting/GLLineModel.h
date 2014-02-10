#include "stdafx.h"
#include "GLViewer.h"
#include "Data3D.h"

namespace GLViewer{ 
	
	class GLLineModel : public GLViewer::Object {
	public:
		Data3D<cv::Vec3f> data; 
		GLLineModel( cv::Vec3i size ) {
			data.resize( size ); 
		}
		virtual ~GLLineModel(void){ }

		virtual void render(void); 

		// size of the object
		virtual unsigned int size_x(void) const { return data.SX(); }
		virtual unsigned int size_y(void) const { return data.SY(); }
		virtual unsigned int size_z(void) const { return data.SZ(); }
	};
}