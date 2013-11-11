#include "MLViwer.h"
// Yuchen: Adding engine for Matlab
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include "engine.h"

#ifdef WIN86
namespace MatlabViwer {
	void plot( Mat_<double> mx, Mat_<double> my )
	{ 
		vector< Mat_<double> > mys( &my, &my+1 );
		plot( mx, mys);
	}

	void plot( Mat_<double> mx, vector< Mat_<double> > mys ){
		for( unsigned int i=0; i<mys.size(); i++ )
		{ 
			if( mx.rows != mys[i].rows ) {
				cout << "Error using ==> plot. " << endl;
				cout << "Vectors must be the same lengths. " << endl;
				return;
			} 
		}
		const int N = mx.rows;

		// open matlab engine
		Engine *ep = engOpen(NULL);
		if ( !ep )
		{
			cout << "Can't start Matlab engine!" <<endl;
			exit(1);
		}

		// The fundamental type underlying MATLAB data
		mxArray *xx = mxCreateDoubleMatrix(1, N, mxREAL); 
		mxArray *yy = mxCreateDoubleMatrix(1, N, mxREAL); 

		// copy data from c++ to matlab data
		memcpy( mxGetPr(xx), mx.data, N * sizeof(double) ); 
		memcpy( mxGetPr(yy), mys[0].data, N * sizeof(double) ); 

		// draw the plot
		engPutVariable( ep, "xx", xx ); 
		engPutVariable( ep, "yy", yy ); 

		engEvalString(ep, "figure(1);");
		engEvalString(ep, "hFig = figure(1);;");
		engEvalString(ep, "set(gcf,'PaperPositionMode','auto');");
		engEvalString(ep, "set(hFig, 'Position', [100 100 900 300]);");
		engEvalString(ep, "hFig = plot(xx, yy);");
		engEvalString(ep, "set(gca, 'XTickLabel', num2str(get(gca,'XTick')','%d'));");
		engEvalString(ep, "set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));");

		// release data
		mxDestroyArray(xx); 
		mxDestroyArray(yy); 

		system("pause");

		// close matlab window
		engClose(ep);
	}

	void surf( Mat_<double> matz ){

		// open matlab engine
		Engine *ep = engOpen(NULL);
		if ( !ep )
		{
			cout << "Can't start Matlab engine!" <<endl;
			exit(1);
		}

		// The fundamental type underlying MATLAB data
		mxArray *z = mxCreateDoubleMatrix(matz.cols, matz.rows, mxREAL); 

		// copy data from c++ to matlab data
		memcpy( mxGetPr(z), matz.data, matz.rows * matz.cols * sizeof(double) ); 

		// draw the plot
		engPutVariable( ep, "z", z ); 

		engEvalString(ep, "axis('axis equal');" );
		engEvalString(ep, "mesh(z);");
		engEvalString(ep, "rotate3d;" );
		engEvalString(ep, "axis('axis equal');" );
		//engEvalString(ep, "figure(1);");
		//engEvalString(ep, "hFig = figure(1);;");
		//engEvalString(ep, "set(gcf,'PaperPositionMode','auto');");
		//engEvalString(ep, "set(hFig, 'Position', [100 100 900 300]);");
		//engEvalString(ep, "hFig = plot(xx, yy);");
		//engEvalString(ep, "set(gca, 'XTickLabel', num2str(get(gca,'XTick')','%d'));");
		//engEvalString(ep, "set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));");

		// release data
		mxDestroyArray(z); 

		system("pause");

		// close matlab window
		engClose(ep);
	}
}
#else 
namespace MatlabViwer {
	void plot( Mat_<double> mx, vector< Mat_<double> > mys ) { }
	void plot( Mat_<double> mx, Mat_<double> my ) { }
	void surf( Mat_<double> matz ) { } 
}
#endif