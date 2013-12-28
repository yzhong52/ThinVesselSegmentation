#include "ImageProcessing.h"
#include "Vesselness.h"
#include <queue>

void ImageProcessing::histogram( Data3D<short>& data, 
	Mat_<double>& range, Mat_<double>& hist, int number_of_bins ){
	cout << "Calculating Image Histogram" << endl;
	
	// get the minimum and maximum value in the data
	Vec<short,2> min_max = data.get_min_max_value();
	const double min = min_max[0];
	const double max = min_max[1] + 1;
	const double diff = max - min;

	// set up the range vector
	range = Mat_<double>( number_of_bins, 1, 0.0);
	for( int i=0; i<number_of_bins; i++ ) {
		range.at<double>(i) = 1.0 * i * diff / number_of_bins + min;
	}
	
	// set the hist vector
	hist = Mat_<double>( number_of_bins, 1, 0.0);
	Mat_<short>::const_iterator it;
	for( it=data.getMat().begin(); it<data.getMat().end(); it++ ) {
		const short& value = (*it);
		int pos = int( 1.0 * number_of_bins * (value-min) / diff );
		hist.at<double>( pos )++;
	}
}

Mat ImageProcessing::histogram_with_opencv( Data3D<short>& data, int number_of_bins ){
	cout << "Calculating Image Histogram using build in OpenCV function" << endl;
	cout << "This function may be fairly memory intensive. Be careful when using." << endl;
	
	Mat mat = data.getMat(); 

	mat.convertTo( mat, CV_32F );
	// normalize( mat, mat, 0, number_of_bins, CV_MINMAX );

	// Establish the number of bins
	int histSize = number_of_bins;

	// Set the ranges
	float range[] = {// 0.0f, number_of_bins
		std::numeric_limits<short>::min(), 
		std::numeric_limits<short>::max() 
	};//the upper boundary is exclusive
	const float* histRange = { range };

	// Compute the histograms:
	Mat hist; 
	bool uniform = true;  // true: bins to have the same size (uniform) 
	bool accumulate = false;
	calcHist( &mat, 1, 0, 
		Mat(), // mask
		hist,  // Output Array
		1, &histSize, &histRange, uniform, true );

	// Draw the histogram
	int hist_w = number_of_bins; 
	int hist_h = int( number_of_bins * 0.618 );
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar(0,0,0) );

	// Normalize the result to [ 0, histImage.rows ]
	normalize( hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound( hist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound( hist.at<float>(i)) ),
			Scalar( 255, 255, 255), 
			2,     // thickness 
			CV_AA, // antialiased line
			0  );
	}

	/// Display
	namedWindow("Histogram", CV_WINDOW_AUTOSIZE );
	imshow("Histogram", histImage );
	imwrite("Histogram.png", histImage );
	waitKey(0);
	cout << "done. " << endl << endl;

	return hist;
}

void ImageProcessing::histogram_for_slice( Image3D<short>& data ){
	cout << "Calculating Image Histogram by Slice" << endl;
	const Mat mat = data.getMat(); 

	vector<double> intensities( data.get_size_z() );
	double max_intensity;
	double min_intensity;
	intensities[0] = max_intensity = min_intensity = cv::sum( mat.row(0) )[0];

	for( int i=1; i<data.get_size_z(); i++ ){
		intensities[i] = cv::sum( mat.row(i) )[0];
		max_intensity = std::max( intensities[i], max_intensity );
		min_intensity = std::min( intensities[i], min_intensity );
	}

	int w = data.get_size_z(); 
	int h = int( data.get_size_z() * 0.618 );

	Mat histImage( h, w, CV_8UC3, Scalar(0,0,0) );
	for( int i=1; i<data.get_size_z(); i++ ){
		line( histImage, 
			Point( i-1, int( h*intensities[i-1]/max_intensity ) ),
			Point( i,   int( h*intensities[i]/max_intensity ) ),
			Scalar( 255, 255, 255), 
			2, // thickness 
			8, // antialiased line
			0  );
	}
	
	namedWindow("Histogram of Slice", CV_WINDOW_AUTOSIZE );
	imshow("Histogram of Slice", histImage );
	imwrite("Histogram of Slice.png", histImage );
	waitKey(0);
}

void ImageProcessing::imNormShow( const string& window_name, const Mat& im ){
	// find the maximum and minimum value of im
	double minVal, maxVal;
	minMaxLoc(im, &minVal, &maxVal);
	Mat dst;
	im.convertTo( dst, CV_32F );
	// Normailize data to [0, 255];
	dst = 255 * ( dst - minVal ) / ( maxVal - minVal );
	dst.convertTo( dst, CV_8U );
	// showing and saving image
	imshow( window_name, dst );
	imwrite( window_name+".bmp", dst );
	cout << "Image is normalized from [" << minVal << ", " << maxVal << "] to [0, 255]" << endl;
}



void ImageProcessing::non_max_suppress( const Data3D<Vesselness_All>& src, Data3D<Vesselness_Sig>& dst ){
	dst.reset( src.get_size() );

	// sqrt(2) and sqrt(3)
	static const float s2 = sqrt(1/2.0f);
	static const float s3 = sqrt(1/3.0f);

	// 13 major orientations in 3d
	static const int MAJOR_DIR_NUM = 13;
	static const Vec3f major_dirs[MAJOR_DIR_NUM] = {
		// directions along the axis, there are 3 of them
		Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0, 0, 1), 
		// directions that lie within the 3D plane of two axis, there are 6 of them
		Vec3f(s2, s2,0), Vec3f(0,s2, s2), Vec3f(s2,0, s2), 
		Vec3f(s2,-s2,0), Vec3f(0,s2,-s2), Vec3f(s2,0,-s2), 
		// directions that are equally in between three axis, there are 4 of them
		Vec3f(s3,s3,s3), Vec3f(s3,s3,-s3), Vec3f(s3,-s3,s3), Vec3f(s3,-s3,-s3)
	};

	// there are 26 directions in 3D
	static const int NUM_DIR_3D = 26;
	Vec3i neighbor3d[NUM_DIR_3D] = {
		// directions along the axis, there are 6 of them
		Vec3i(0,0, 1), Vec3i(0, 1,0),  Vec3i( 1,0,0),
		Vec3i(0,0,-1), Vec3i(0,-1,0),  Vec3i(-1,0,0), 
		// directions that lie within the 3D plane of two axis, there are 12 of them
		Vec3i(0,1,1), Vec3i(0, 1,-1), Vec3i( 0,-1,1), Vec3i( 0,-1,-1), 
		Vec3i(1,0,1), Vec3i(1, 0,-1), Vec3i(-1, 0,1), Vec3i(-1, 0,-1), 
		Vec3i(1,1,0), Vec3i(1,-1, 0), Vec3i(-1, 1,0), Vec3i(-1,-1, 0), 
		// directions that are equally in between three axis, there are 8 of them
		Vec3i( 1,1,1), Vec3i( 1,1,-1), Vec3i( 1,-1,1), Vec3i( 1,-1,-1), 
		Vec3i(-1,1,1), Vec3i(-1,1,-1), Vec3i(-1,-1,1), Vec3i(-1,-1,-1), 
	};

	// cross section for the major orientation
	vector<Vec3i> cross_section[MAJOR_DIR_NUM];
	
	// Setting offsets that are perpendicular to dirs
	for( int i=0; i<MAJOR_DIR_NUM; i++ ) {
		for( int j=0; j<NUM_DIR_3D; j++ ){
			// multiply the two directions
			float temp = major_dirs[i].dot( neighbor3d[j] );
			// the temp is 0, store this direciton
			if( abs(temp)<1.0e-5 ) {
				cross_section[i].push_back( neighbor3d[j] );
			}
		}
	}

	int x,y,z;
	for( z=0; z<src.get_size_z(); z++ ) {
		for( y=0; y<src.get_size_y(); y++ ) {
			for( x=0; x<src.get_size_x(); x++ ) {
				// non-maximum surpression
				bool isMaximum = true;

				// Method I: 
				// find the major orientation
				// assigning the orientation to one of the 13 categories
				const Vec3f& cur_dir = src.at(x,y,z).dir;
				int mdi = 0; // major direction id
				float max_dot_product = 0;
				for( int di=0; di<MAJOR_DIR_NUM; di++ ){
					float current_dot_product = abs( cur_dir.dot(major_dirs[di]) );
					if( max_dot_product < current_dot_product ) { 
						max_dot_product = current_dot_product;
						mdi = di;// update the major direction id
					}
				}
				for( unsigned int i=0; i<cross_section[mdi].size(); i++ ){
					int ox = x + cross_section[mdi][i][0];
					int oy = y + cross_section[mdi][i][1];
					int oz = z + cross_section[mdi][i][2];
					if( src.isValid(ox,oy,oz) && src.at(x,y,z).rsp < src.at(ox,oy,oz).rsp ) {
						isMaximum = false; break;
					} 
				}

				// Method II: Based on Sigma
				//int sigma = (int) ceil( src.at(x,y,z).sigma );
				//for( int i = -sigma; i <= sigma; i++ ) for( int j = -sigma; j <= sigma; j++ ) {
				//	Vec3i offset = src.at(x,y,z).normals[0] * i + src.at(x,y,z).normals[1] * j;
				//	int ox = x + offset[0];
				//	int oy = y + offset[1];
				//	int oz = z + offset[2];
				//	if( src.isValid(ox,oy,oz) && src.at(x,y,z).rsp < src.at(ox,oy,oz).rsp ) {
				//		isMaximum = false; break;
				//	} 
				//}

				if( isMaximum ) {
					dst.at(x,y,z) = src.at(x,y,z);
					// dst.at(x,y,z).rsp = sqrt( dst.at(x,y,z).rsp ); 
				}
			}
		}
	}
}



void ImageProcessing::edge_tracing( Data3D<Vesselness_Sig>& src, Data3D<Vesselness_Sig>& dst, const float& thres1, const float& thres2 )
{
	Data3D<float> src1d;
	src.copyDimTo( src1d, 0 );
	IP::normalize( src1d, 1.0f );
	
	int x, y, z;
	std::queue<Vec3i> q;
	Data3D<unsigned char> mask( src.get_size() );
	for(z=0;z<src.SZ();z++) for (y=0;y<src.SY();y++) for(x=0;x<src.SX();x++) {
		if( src1d.at(x,y,z) > thres1 ) {
			q.push( Vec3i(x,y,z) );
			mask.at(x,y,z) = 255;
		}
	} 

	Vec3i dir[26];
	for( int i=0; i<26; i++ ){
		int index = (i + 14) % 27;
		dir[i][0] = index/9%3 - 1;
		dir[i][1] = index/3%3 - 1;
		dir[i][2] = index/1%3 - 1;
	}

	while( !q.empty() ){
		Vec3i pos = q.front(); q.pop();
		Vec3i off_pos;
		for( int i=0; i<26; i++ ) {
			off_pos = pos + dir[i];
			if( src.isValid(off_pos) && !mask.at( off_pos ) && src1d.at(off_pos) > thres2 ){
				mask.at( off_pos ) = 255;
				q.push( off_pos );
			}
		}
	}

	dst.reset( src.get_size() );
	for(z=0;z<src.SZ();z++) for (y=0;y<src.SY();y++) for(x=0;x<src.SX();x++) {
		if( mask.at( x,y,z ) ) {
			dst.at( x,y,z ) = src.at( x,y,z );
			// dst.at( x,y,z ).rsp = sqrt( src1d.at( x,y,z ) );
		}
	} 
}

void ImageProcessing::dir_tracing( Data3D<Vesselness_All>& src_vn, Data3D<Vesselness_Sig>& dst ){
	// non-maximum suppression
	Data3D<Vesselness_Sig> res_nms; 
	IP::non_max_suppress( src_vn, res_nms );

	float thres1 = 0.800f;
	float thres2 = 0.10f;

	Vec3i offset[26];
	for( int i=0; i<26; i++ ){
		int index = (i + 14) % 27;
		offset[i][0] = index/9%3 - 1;
		offset[i][1] = index/3%3 - 1;
		offset[i][2] = index/1%3 - 1;
	}

	Data3D<float> src1d;
	res_nms.copyDimTo( src1d, 0 );
	IP::normalize( src1d, 1.0f );
	
	// find seed points
	int x, y, z;
	std::queue<Vec3i> q;
	Data3D<unsigned char> mask( res_nms.get_size() );
	for(z=0;z<res_nms.SZ();z++) for (y=0;y<res_nms.SY();y++) for(x=0;x<res_nms.SX();x++) {
		if( src1d.at(x,y,z) > thres1 ) {
			q.push( Vec3i(x,y,z) );
			mask.at(x,y,z) = 255;
		}
	} 

	// region growing
	while( !q.empty() ){
		Vec3i pos = q.front(); q.pop();
		Vec3i off_pos;
		for( int i=0; i<26; i++ ) {
			off_pos = pos + offset[i];
			if( res_nms.isValid(off_pos) && !mask.at( off_pos ) && src1d.at(off_pos) > thres2 ){
				mask.at( off_pos ) = 255;
				q.push( off_pos );
			}
		}
	}
	
	
	dst.reset( src_vn.get_size() );
	for(z=0;z<src_vn.SZ();z++) for (y=0;y<src_vn.SY();y++) for(x=0;x<src_vn.SX();x++) {
		if( mask.at( x,y,z ) ) {

			int count = 0;
			int index = 0;
			for( int i=0; i<26; i++ ){ 
				Vec3i pos = Vec3i(x,y,z) + offset[i];
				if( mask.isValid(pos) && mask.at(pos) ) {
					count++;
					index = i;
				}
			}
			if( count==1 ) { 
				Vec3i bPos[20];
				// trace the gap bettween bifurcations
				Vec3f dir = src_vn.at(x,y,z).dir;
				if( dir.dot( offset[index] ) > 0 ) dir = -dir;
				
				Vec3f pos(1.0f*x, 1.0f*y, 1.0f*z);
				pos += dir;
				for( int i=1; i<20; i++ ){
					pos += dir;
					bPos[i] = Vec3i( pos );
					if( !mask.isValid( bPos[i] ) ) break;

					bool flag = false; 
					for( int j=0; j<26; j++ ) {
						Vec3f off_pos = pos;
						off_pos += offset[j];
						if( mask.isValid(off_pos) && mask.at( off_pos ) ) {
							flag = true; break;	
						}
					}

					if( flag == true ) {
						// back tracing
						while( --i ) {
							dst.at( bPos[i] ) = src_vn.at( bPos[i] );
							dst.at( bPos[i] ).rsp = 1.0f; sqrt( src1d.at( bPos[i] ) );
						}
						break;
					}
				}
			}
			
			dst.at( x,y,z ) = src_vn.at( x,y,z );
			dst.at( x,y,z ).rsp = sqrt( src1d.at( x,y,z ) );
		}
	} 
	
	return;
}




void ImageProcessing::edge_tracing_mst( Data3D<Vesselness_All>& src_vn, Data3D<Vesselness_Sig>& dst, const float& thres1, const float& thres2  ){
	dst.reset( src_vn.get_size() );

	// non-maximum suppression
	Data3D<Vesselness_Sig> res_nms; 
	IP::non_max_suppress( src_vn, res_nms );
	
	Vec3i offset[26];
	for( int i=0; i<26; i++ ){
		int index = (i + 14) % 27;
		offset[i][0] = index/9%3 - 1;
		offset[i][1] = index/3%3 - 1;
		offset[i][2] = index/1%3 - 1;
	}

	// normailize the data based on vesselness response
	Data3D<float> src1d;
	res_nms.copyDimTo( src1d, 0 );
	IP::normalize( src1d, 1.0f );

	// find seed points with the 1st threshold
	int x, y, z;
	std::queue<Vec3i> myQueue1;
	Data3D<unsigned char> mask( res_nms.get_size() );
	for(z=0;z<res_nms.SZ();z++) for (y=0;y<res_nms.SY();y++) for(x=0;x<res_nms.SX();x++) {
		if( src1d.at(x,y,z) > thres1 ) {
			myQueue1.push( Vec3i(x,y,z) );
			mask.at(x,y,z) = 255;
		}
	} 
	// region growing base on the second threshold
	while( !myQueue1.empty() ){
		Vec3i pos = myQueue1.front(); myQueue1.pop();
		Vec3i off_pos;
		for( int i=0; i<26; i++ ) {
			off_pos = pos + offset[i];
			if( res_nms.isValid(off_pos) && !mask.at( off_pos ) && src1d.at(off_pos) > thres2 ){
				mask.at( off_pos ) = 255;
				myQueue1.push( off_pos );
			}
		}
	}

	// Now we have the mask of the center-line. We want to connect them as a semi minimum spinning tree. 
	// But before that, we need to know their conectivity. We achieve this by labeling them with a setid. 
	// If two voxel have the same setid, they are connected. 
	Data3D<unsigned char> setid( mask.get_size() );
	int max_sid = 0; // the maximum set id that being used 
	for(z=0;z<src_vn.SZ();z++) for (y=0;y<src_vn.SY();y++) for(x=0;x<src_vn.SX();x++) {
		// if this voxel belongs to the center-line and has not been labeled yet.
		// We will run a breadth-first search from this voxel to label all other labels that are connected to this one. 
		if( mask.at(x,y,z)==255 && setid.at(x,y,z)==0 ){ 
			setid.at(x,y,z) = ++max_sid; 
			// breadth-first search
			std::queue<Vec3i> q;
			q.push( Vec3i(x,y,z) );
			while( !q.empty() ){
				Vec3i pos = q.front(); q.pop();
				for( int i=0; i<26; i++ ) {
					Vec3i off_pos = pos + offset[i];
					if( mask.isValid( off_pos) && mask.at( off_pos )==255 && setid.at( off_pos )!=max_sid ) {
						setid.at( off_pos )=max_sid;
						q.push( off_pos );
					} 
				}
			}
		}
	}
	cout << "Max sid: " <<  max_sid << endl;
	
	const unsigned char ENDPOINT_YES = 255;
	const unsigned char ENDPOINT_NO  = 144;
	const unsigned char UN_DEFINED  = 0;   // undefined
	Data3D<unsigned char> endpoints_mask1( mask.get_size() );
	for(z=1;z<src_vn.SZ()-1;z++) for (y=1;y<src_vn.SY()-1;y++) for(x=1;x<src_vn.SX()-1;x++) {
		if( mask.at( x,y,z )==0 ) continue;
		if( endpoints_mask1.at(x,y,z)!=UN_DEFINED ) continue;
		// breath-first search begin
		std::queue<Vec3i> myQueue;
		myQueue.push( Vec3i(x,y,z) );
		while( !myQueue.empty() ) {
			Vec3i pos = myQueue.front(); myQueue.pop();
			// initial guess the this pos to be a endpoint
			endpoints_mask1.at( pos ) = ENDPOINT_YES;
			// transverse the neighbour hood system
			for( int i=0; i<26; i++ ) { 
				Vec3i off_pos = pos + offset[i];
				if( !mask.isValid( off_pos ) ) continue;
				if( mask.at(off_pos)==0 ) continue;
				if( endpoints_mask1.at(off_pos)==UN_DEFINED ) {
					myQueue.push( off_pos );
					endpoints_mask1.at( off_pos ) = ENDPOINT_YES; 
					endpoints_mask1.at( pos ) = ENDPOINT_NO;
				} else if( endpoints_mask1.at(off_pos)==ENDPOINT_YES ) { 
					endpoints_mask1.at( pos ) = ENDPOINT_NO;
				}
			}
		}
	}
	Data3D<unsigned char> endpoints_mask2( mask.get_size() );
	for(z=src_vn.SZ()-2;z>=1;z--) for(y=src_vn.SY()-2;y>=1;y--) for(x=src_vn.SX()-2; x>=1; x--) {
		if( mask.at( x,y,z )==0 ) continue;
		if( endpoints_mask2.at(x,y,z)!=UN_DEFINED ) continue;
		// breath-first search begin
		std::queue<Vec3i> myQueue;
		myQueue.push( Vec3i(x,y,z) );
		while( !myQueue.empty() ) {
			Vec3i pos = myQueue.front(); myQueue.pop();
			// initial guess the this pos to be a endpoint
			endpoints_mask2.at( pos ) = ENDPOINT_YES;
			// transverse the neighbour hood system
			for( int i=0; i<26; i++ ) { 
				Vec3i off_pos = pos + offset[i];
				if( !mask.isValid( off_pos ) ) continue;
				if( mask.at(off_pos)==0 ) continue;
				if( endpoints_mask2.at(off_pos)==UN_DEFINED ) {
					myQueue.push( off_pos );
					endpoints_mask2.at( off_pos ) = ENDPOINT_YES; 
					endpoints_mask2.at( pos ) = ENDPOINT_NO;
				} else if( endpoints_mask2.at(off_pos)==ENDPOINT_YES ) { 
					endpoints_mask2.at( pos ) = ENDPOINT_NO;
				}
			}
		}
	}

	vector<Vec3i> endpoints; // endpoints are the points that have only one neighbour
	for(z=1;z<mask.SZ()-1;z++) for (y=1;y<mask.SY()-1;y++) for(x=1;x<mask.SX()-1;x++) {
		if( endpoints_mask1.at(x,y,z)==ENDPOINT_YES || endpoints_mask2.at(x,y,z)==ENDPOINT_YES ) {
			if( src1d.at(x,y,z)> (0.5f*thres1+0.5f*thres2) ) {
				endpoints.push_back( Vec3i(x,y,z) );
			}
		}
	} 

	//// For visualization of the end points
	//for(z=0;z<mask.SZ();z++) for (y=0;y<mask.SY();y++) for(x=0;x<mask.SX();x++) {
	//	if( mask.at(x,y,z)==255) dst.at(x,y,z).rsp = 0.35f;
	//	if( endpoints_mask1.at(x,y,z)==ENDPOINT_YES || endpoints_mask2.at(x,y,z)==ENDPOINT_YES ) {
	//		if( src1d.at(x,y,z)> (0.5f*thres1+0.5f*thres2) ) {
	//			dst.at(x,y,z).rsp = 1.0f;
	//		}
	//	}
	//} 
	//return;

	// two small data structures for the use priority_queue
	class Dis_Pos {
	private: 
		float dist;
		Vec3i to_pos;
	public:
		Dis_Pos( const float& distance, const Vec3i& position ) 
			: dist(distance) 
			, to_pos(position) { }
		inline bool operator<( const Dis_Pos& right ) const { 
			// for the use of priority_queue, we reverse the sign of comparison from '<' to '>'
			return ( this->getDist() > right.getDist() ); 
		} 
		inline const float& getDist(void) const { return dist; }
		inline const Vec3i& getToPos(void) const { return to_pos; }
	};
	class Dis_Pos_Pos : public Dis_Pos {
	private:
		Vec3i from_pos;
	public:
		Dis_Pos_Pos( const Dis_Pos& dis_pos, const Vec3i& from_posistion ) 
			: Dis_Pos(dis_pos), from_pos(from_posistion) { }
		inline const Vec3i& getFromPos(void) const { return from_pos; }
	};


	std::priority_queue< Dis_Pos_Pos > min_dis_queue; 
	const unsigned char VISITED_YES = 255;
	const unsigned char VISITED_N0  = 0;
	Data3D<unsigned char> isVisited( setid.get_size() );
	vector<Vec3i>::iterator it;
	for( it=endpoints.begin(); it<endpoints.end(); it++ ) {
		const Vec3i& from_pos = *it;
		std::priority_queue< Dis_Pos > myQueue; 
		myQueue.push( Dis_Pos( 0.0f, from_pos) );
		
		// breath first search
		isVisited.reset(); // set all the data to 0. 
		isVisited.at( from_pos ) = 255;
		bool to_pos_fount = false; 
		while( !myQueue.empty() && !to_pos_fount ) {
			Dis_Pos dis_pos = myQueue.top(); myQueue.pop(); 
			for( int i=0; i<26; i++ ) { 
				// get the propogate position
				Vec3i to_pos = offset[i] + dis_pos.getToPos();
				if( !isVisited.isValid(to_pos) ) continue;
				if(  isVisited.at(to_pos)==VISITED_YES ) continue;
				
				if( setid.at(to_pos)!=setid.at(from_pos) ) {
					Vec3i dif = to_pos - from_pos;
					float dist = sqrt( 1.0f*dif[0]*dif[0] + dif[1]*dif[1] + dif[2]*dif[2] );
					if( dist > 7.0f ) continue; // prevent from searching too far aways

					if( setid.at( to_pos )==0 || src1d.at(to_pos)< thres1 ) {	
						// if this voxel belongs to a background set
						myQueue.push( Dis_Pos(dist, to_pos) );
						isVisited.at( to_pos ) = VISITED_YES;
					}
					
					else {
						// if this voxels have different setid
						to_pos_fount = true;
						float ratio = src1d.at(from_pos) / src1d.at(to_pos);
						if( ratio<1.0f ) ratio = 1 / ratio;
						dist += ratio;
						min_dis_queue.push( Dis_Pos_Pos(Dis_Pos(dist*ratio, to_pos), from_pos) );
						break;
					}
				}
			}
		}
	}

	while( !min_dis_queue.empty() ){
		Dis_Pos_Pos dpp = min_dis_queue.top(); min_dis_queue.pop();
		const Vec3i& from_pos = dpp.getFromPos();
		const Vec3i& to_pos = dpp.getToPos();
		const float& dist = dpp.getDist(); 
		if( setid.at(to_pos)==setid.at(from_pos) ) continue;

		// And we will use breadth first search one more time to make sure the setid of 
		// to_pos is the same as from_pos. 
		std::queue< Vec3i > myQueue; 
		myQueue.push( to_pos );
		int to_setid = setid.at(to_pos); 
		setid.at(to_pos) = setid.at( from_pos );
		while( !myQueue.empty() ){
			Vec3i pos = myQueue.front(); myQueue.pop();
			for( int i=0; i<26; i++ ) { 
				Vec3i off_pos = pos + offset[i];
				if( setid.isValid(off_pos) && setid.at(off_pos)==to_setid ){
					myQueue.push( off_pos );
					setid.at(off_pos) = setid.at(from_pos);
				}
			}
		}

		// connect from_pos and to_pos
		Vec3f dir = to_pos - from_pos;
		dir[0] /= dist;
		dir[1] /= dist;
		dir[2] /= dist;
		Vec3f pos = from_pos;
		pos += dir;
		for( int i=1; i<dist; i++ ){
			dst.at( Vec3i(pos) ).rsp = 1.0f;
			setid.at(pos) = setid.at(from_pos);
			pos+=dir;
		}
		
	}

	for(z=0;z<src_vn.SZ();z++) for (y=0;y<src_vn.SY();y++) for(x=0;x<src_vn.SX();x++) {
		if( mask.at(x,y,z)==255) dst.at(x,y,z).rsp = sqrt( src1d.at(x,y,z) );
	}
	
	return;
}



void ImageProcessing::dilation( Data3D<unsigned char>& src, const int& ks ){
	Data3D<unsigned char> temp( src.get_size() );
	int x,y,z;
	int i,j,k;
	for( z=0;z<src.SZ();z++ ) for( y=0;y<src.SY();y++ ) for( x=0;x<src.SX();x++ ) {
		// look around the voxel
		for( i=-ks;i<=ks;i++ ) for( j=-ks;j<=ks;j++ ) for( k=-ks;k<=ks;k++ ) {
			if( src.isValid(x+i,y+j,z+k) ) {
				if( src.at( x+i,y+j,z+k )!=0 ) temp.at(x,y,z) = 255; 
			}
		}
	}
	src = temp;
}


void ImageProcessing::erosion( Data3D<unsigned char>& src, const int& ks ){
	Data3D<unsigned char> temp( src.get_size(), 255 );
	int x,y,z;
	int i,j,k;
	for( z=0;z<src.SZ();z++ ) for( y=0;y<src.SY();y++ ) for( x=0;x<src.SX();x++ ) {
		// look around the voxel
		for( i=-ks;i<=ks;i++ ) for( j=-ks;j<=ks;j++ ) for( k=-ks;k<=ks;k++ ) {
			if( src.isValid(x+i,y+j,z+k) ){
				if( src.at( x+i,y+j,z+k )!=255 ) temp.at(x,y,z) =  0; 
			} else {
				temp.at(x,y,z) = 0;
			}
		}
	}
	src = temp;
}

void ImageProcessing::closing( Data3D<unsigned char>& src, const int& ks ){
	IP::dilation( src, 2 );
	IP::erosion( src, 2 );
}