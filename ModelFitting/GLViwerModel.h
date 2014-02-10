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
	void addModel( GLViewer::GLLineModel* lineModel );
};

