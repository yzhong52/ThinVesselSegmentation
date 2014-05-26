#include "init_models.h"
#include "VesselnessTypes.h"
#include "Line3DTwoPoint.h"
#include "ModelSet.h"
#include "Data3D.h"
#include "ImageProcessing.h"
#include "Neighbour26.h"

void each_model_per_point(
	const Data3D<Vesselness_Sig>& vn_sig,
	Data3D<int>& labelID3d,
	std::vector<cv::Vec3i>& tildaP,
	ModelSet<Line3D>& model,
	std::vector<int>& labelID,
	const float& threshold )
{
	Data3D<float> vn = vn_sig;
	IP::normalize( vn, 1.0f );

	tildaP.clear();
	labelID.clear();
	labelID3d.reset( vn.get_size(), -1 );
	model.models.clear();

	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
		if( vn.at(x,y,z) > threshold ) { // a thread hold
			int lid = (int) model.models.size();

			labelID3d.at(x,y,z) = lid;

			labelID.push_back( lid );

			tildaP.push_back( Vec3i(x,y,z) );

			const Vec3d pos(x,y,z);
			const Vec3d& dir = vn_sig.at(x,y,z).dir;
			const double& sigma = vn_sig.at(x,y,z).sigma;
			Line3DTwoPoint *line  = new Line3DTwoPoint();
			line->setPositions( pos-dir, pos+dir );
			line->setSigma( sigma );
			model.models.push_back( line );
		}
	}
}


void each_model_per_local_maximum(
	const Data3D<Vesselness_Sig>& vn_sig,
	Data3D<int>& labelID3d,
	vector<cv::Vec3i>& tildaP,
	ModelSet<Line3D>& model,
	vector<int>& labelID )
{
	Data3D<float> vn = vn_sig;
	IP::normalize( vn, 1.0f );

	Data3D<Vec3i> djs( vn.get_size() );

	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
		// find the major orientation
		// assigning the orientation to one of the 13 categories
		const Vec3f& cur_dir = vn_sig.at(x,y,z).dir;
		int mdi = 0; // major direction id
		float max_dot_product = 0;
		for( int di=0; di<13; di++ ){
			float current_dot_product = cur_dir.dot( Neighbour26::at(di) );
			if( max_dot_product < current_dot_product ) {
				max_dot_product = current_dot_product;
				mdi = di;// update the major direction id
			}
		}
		const vector<cv::Vec3i>& cross_section = Neighbour26::getCrossSection(mdi);
		Vec3i mics( -1, -1, -1 ); // maximum index on cross section
		float mrsp = vn.at(x,y,z); // maximum response
		for( unsigned i =0; i < cross_section.size(); i++ ){
			Vec3i offset_pos  = Vec3i(x,y,z) + cross_section[i];
			if( vn.isValid( offset_pos ) && mrsp < vn.at( offset_pos ) ) {
				mrsp = vn.at(offset_pos);
				mics = offset_pos;
			}
		}
		// 'mics==Vec3i( -1, -1, -1 )' indicates that the current position is local maximum
		djs.at(x,y,z) = mics;
	}


	tildaP.clear();
	labelID.clear();
	labelID3d.reset( vn.get_size(), -1 );
	model.models.clear();
	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
		if( vn.at(x,y,z) > 0.1f && djs.at(x,y,z)==Vec3i( -1, -1, -1 ) ) {
			int lid = (int) model.models.size();

			labelID3d.at(x,y,z) = lid;

			labelID.push_back( lid );

			tildaP.push_back( Vec3i(x,y,z) );

			const Vec3d pos(x,y,z);
			const Vec3d& dir = vn_sig.at(x,y,z).dir;
			const double& sigma = vn_sig.at(x,y,z).sigma;
			Line3DTwoPoint *line  = new Line3DTwoPoint();
			line->setPositions( pos-dir, pos+dir );
			line->setSigma( sigma );
			model.models.push_back( line );
		}
	}

	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
		Vec3i pos(x,y,z);
		while( djs.at( pos )!=Vec3i( -1, -1, -1 ) ) {
			pos = djs.at( pos );
		}
		if( labelID3d.at(pos)!=-1 ) {
			labelID3d.at(x,y,z) = labelID3d.at(pos);
			tildaP.push_back( Vec3i(x,y,z) );
			labelID.push_back( labelID3d.at(pos) );
		}
	}
}
