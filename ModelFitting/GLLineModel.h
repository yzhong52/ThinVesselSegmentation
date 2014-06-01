#ifndef GLLINEMODEL_H_
#define GLLINEMODEL_H_

#include "GLViewer.h"
#include "Data3D.h"
#include "Line3D.h"
#include <vector>

using namespace std;
using namespace cv;


namespace GLViewer{
	// This class is thread safe
	class GLLineModel : public GLViewer::Object {
		cv::Vec3i size;

		// The following two vectors have same sizes
		// A vector of the points to lines
		vector<Line3D*> lines;
		// a vector of the colors for the lines
		vector<Vec3b> lineColors; // 3 unsigned char

		// The following two vectors have same sizes
		vector<cv::Vec3i> dataPoints;
		vector<int> labelings;

		char render_mode;
	public:
		GLLineModel( cv::Vec3i size );

		virtual ~GLLineModel( void );

		virtual void render( void );

		void updatePoints( const vector<Vec3i>& pts );
		void updateModel( const vector<Line3D*>& lns, const vector<int>& lbls );
		// size of the object
		virtual unsigned int size_x(void) const { return size[0]; }
		virtual unsigned int size_y(void) const { return size[1]; }
		virtual unsigned int size_z(void) const { return size[2]; }

		void init(void);

		virtual void keyboard( unsigned char key );
	};
}

#endif
