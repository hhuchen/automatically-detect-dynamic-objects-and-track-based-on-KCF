#ifndef process_cl
#define process_cl

#include "opencv2/core/core.hpp"
#include <vector>
#include <list>
#include <opencv/cv.h>

using namespace cv;

struct num_list
{
	int num1;
	int num2;
	int num3;
	int num4;
};

class processimg_cl
{
public:
	int cs;
	int counts;
	int a[4];
	num_list cl_need[100];
	Mat result;
	int match_method;
	char* image_window; //窗口名称定义
	char* result_window;  //窗口名称定义
	Point minloc_get_first;

	processimg_cl();
	
protected:

private:

};

class ExtractorNode
{
public:
	ExtractorNode():bNoMore(false){}

	void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

	std::vector<cv::KeyPoint> vKeys;
	cv::Point2i UL, UR, BL, BR;
	std::list<ExtractorNode>::iterator lit;
	bool bNoMore;
};


class ORBextractor
{
public:

	enum {HARRIS_SCORE=0, FAST_SCORE=1 };

	ORBextractor(int features_num = 500, float scale_factor = 1.2f, int levels_num = 8,
		int default_fast_threshold = 20, int min_fast_threshold = 7);

	~ORBextractor(){}


	// Compute the ORB features and descriptors on an image.
	// ORB are dispersed on the image using an octree.
	// Mask is ignored in the current implementation.
	void operator()( cv::InputArray image,
		std::vector<cv::KeyPoint>& keypoints);

	int inline GetLevels(){
		return nlevels;}

	float inline GetScaleFactor(){
		return scaleFactor;}

	std::vector<float> inline GetScaleFactors(){
		return mvScaleFactor;
	}

	std::vector<float> inline GetInverseScaleFactors(){
		return mvInvScaleFactor;
	}

	std::vector<float> inline GetScaleSigmaSquares(){
		return mvLevelSigma2;
	}

	std::vector<float> inline GetInverseScaleSigmaSquares(){
		return mvInvLevelSigma2;
	}

	std::vector<cv::Mat> mvImagePyramid;

protected:

	void ComputePyramid(cv::Mat image);
	void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
	std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
		const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

	void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
	std::vector<cv::Point> pattern;

	int nfeatures;
	double scaleFactor;
	int nlevels;
	int iniThFAST;
	int minThFAST;

	std::vector<int> mnFeaturesPerLevel;

	std::vector<int> umax;

	std::vector<float> mvScaleFactor;
	std::vector<float> mvInvScaleFactor;    
	std::vector<float> mvLevelSigma2;
	std::vector<float> mvInvLevelSigma2;
};





#endif