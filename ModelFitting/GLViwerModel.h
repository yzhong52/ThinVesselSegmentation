#pragma once

#include "GLViwerWrapper.h"
#include "GLLineModel.h"

#include "gco-v3.0\GCoptimization.h"

class Line3D;

class GLViwerModel : public GLViewerExt
{
public:
	GLViwerModel(void);
	virtual ~GLViwerModel(void);

	void addModel( GCoptimization* ptrGC, vector<Line3D> lines, Vec3i size );
	
	// Visulizing Ryen's Data
	void GLViwerModel::addModel(
		vector<Line3D>& lines, // the labels
		vector<vector<Vec3i> >& pointsSet, // there corresponding points
		cv::Vec3i& size ); 
	void addModel( GLViewer::GLLineModel* lineModel );
};

