// get_camera.cpp : 定义控制台应用程序的入口点。
//


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "highgui/highgui.hpp"  
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/core/core.hpp>


#include <string>
#include <iostream>


#include "processimg.h"
#include "kcftracker.hpp"

#define IS_REFINE_FUNDA 1
#define IS_REFINE_MATCHES 1

using namespace cv;
using namespace std;

bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;

KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
// Tracker results
Rect result;

struct Frame
{
	cv::Mat frame;
	int frame_id;
	std::vector<cv::KeyPoint> cv_feats;  //特征点
	cv::Mat                   cv_descs;  //特征描述子
	vector<Point2f>           corners;   //KLT特征点
};


struct Initial_points
{
	vector<Point2f> initial;//提取出的KLT特征点
	vector<Point2f> cur_corners;    // 初始化跟踪点的位置

};

//variable
std::vector<cv::KeyPoint> cv_feats;  //特征点
cv::Mat                   cv_descs;  //特征描述子

int frame_id = 0;

Frame last_frame, cur_frame;
Mat initial_img;

vector<Point2f> points[2];  // point0为特征点的原来位置，point1为特征点的新位置
vector<Point2f> corners;//提取出的KLT特征点
vector<Point2f> initial;    // 初始化跟踪点的位置

vector<uchar> status;   // 跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;

bool flag_start_track = false;//检测动态物体成功进行跟踪物体的标志位

//function
void ExtractORB(const cv::Mat img);   
void Process(cv::Mat im);
bool acceptTrackedPoint(int i);
Initial_points detection_initial(Mat initial, Mat cur, vector<Point2f> first, vector<Point2f> next);
vector<Point2f> detection_modle(Mat initial,Mat cur,vector<Point2f> first, vector<Point2f> next,Mat img);
vector<Point2f> epipolar_modle(vector<Point2f> points,Mat F);

int main(int argc, char** argv)
{
	VideoCapture inputVideo(0);

	if (!inputVideo.isOpened())
	{
		cout << "Could not open the input video " << endl;
		return -1;
	}
	Mat frame;
	string imgname;
	int f = 1;
	
	while (1) //Show the image captured in the window and repeat
	{
		inputVideo >> frame;              // read
		if (frame.empty()) break;         // check if at end
		frame_id++;
		imshow("img", frame);
		//cv::cvtColor(frame,frame,CV_BGR2GRAY);
		if (frame_id>10)
		{
			Process(frame);
		}
		
		//ExtractORB(frame);

		char key = waitKey(1);
		if (key == 27)break;
		if (key == 'q' || key == 'Q')
		{
			imgname = to_string(f++) + ".jpg";
			imwrite(imgname, frame);
			cout << "Finished writing" << endl;
		}
	}
	return 0;
}

void ExtractORB(const cv::Mat img)
{
	Mat imageSource;
	
	imageSource = img.clone();
	goodFeaturesToTrack(imageSource, corners, 500, 0.01, 10, Mat());
	cout << "提取的特征点数为：" << corners.size() << endl;
	cv::cvtColor(imageSource, imageSource, CV_GRAY2RGB);
	for (int i = 0; i<corners.size(); i++)
	{
		circle(imageSource, corners[i], 1, Scalar(0, 255, 0), 2);
	}
	//corners.clear();
	imshow("Corner Detected", imageSource);


	//ORBextractor extractor;
	//extractor(img, cv_feats);
	//cv::Mat img_rgb = cv::Mat(img.size(), CV_8UC3);
	//cv::cvtColor(img, img_rgb, CV_GRAY2RGB);
	///*std::for_each(cv_feats.begin(), cv_feats.end(), [&](cv::KeyPoint i) {
	//	cv::circle(img_rgb, i.pt, 2, cv::Scalar(0, 255, 0), 1);
	//});*/
	//imshow("ORB特征点", img_rgb);
	/*cv::Mat im;
	pyrUp(img, im, Size(im.cols * 2, im.rows * 2)); //放大一倍
	Ptr<ORB> orb = ORB::create(1500);
	orb->detect(im,cv_feats);
	orb->compute(im,cv_feats,cv_descs);
	cv::Mat img_opencv = cv::Mat(im.size(), CV_8UC3);
	cv::cvtColor(im, img_opencv, CV_GRAY2RGB);
	std::for_each(cv_feats.begin(), cv_feats.end(), [&](cv::KeyPoint i) {
		cv::circle(img_opencv, i.pt, 4 * (i.octave + 1), cv::Scalar(0, 255, 0), 1);
	});
	namedWindow("ORB特征点", 0);
	imshow("ORB特征点", img_opencv);*/
}



//对提取出的角点做一定的预处理
void DlimiteOrb(vector<Point2f>& inigoodfeatures)
{
	vector<Point2f> newgoodfeatures;
	Point2f temp1, temp2;
	for (size_t i = 0; i<inigoodfeatures.size(); i++)
	{
		temp1 = inigoodfeatures.at(i);
		for (size_t k = 0; k<i; k++)
			newgoodfeatures.push_back(inigoodfeatures.at(k));
		newgoodfeatures.push_back(temp1);
		for (size_t j = i + 1; j<inigoodfeatures.size(); j++)
		{
			temp2 = inigoodfeatures.at(j);
			if (abs(temp1.x - temp2.x) + abs(temp1.y - temp2.y)>10)
				newgoodfeatures.push_back(temp2);
		}
		inigoodfeatures = newgoodfeatures;
		newgoodfeatures.clear();
	}
}


//  检测新点是否应该被添加
// return: 是否被添加标志
bool addNewPoints()
{
	return last_frame.corners.size() <= 10;
}

void Process(cv::Mat im)
{
	if (addNewPoints())
	{
		initial_img = im.clone();
		last_frame.frame = im.clone();
		last_frame.frame_id = frame_id;

		//ORBextractor extractor;
		//extractor(im, cv_feats);
		cv::cvtColor(last_frame.frame, last_frame.frame,CV_BGR2GRAY);

		goodFeaturesToTrack(last_frame.frame, corners, 500, 0.01, 5, Mat());
		//KeyPoint::convert(cv_feats,corners);
		
		cout << "提取的特征点数为：" <<corners.size() << endl;
		last_frame.corners = corners;
		initial = corners;
		//cv::cvtColor(im, im, CV_GRAY2RGB);
		for (int i = 0; i<last_frame.corners.size(); i++)
		{
			circle(im, last_frame.corners[i], 1, Scalar(0, 0, 255), 2);
		}
		imshow("Corner Detect", im);
	}
	else
	{
		cur_frame.frame = im.clone();
		Mat imgsource = im.clone();
		cur_frame.frame_id = frame_id;

		cv::cvtColor(cur_frame.frame, cur_frame.frame, CV_BGR2GRAY);

		//跟踪程序
		TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
		double derivlambda = 0.5;
		int flags = 0;

		cv::calcOpticalFlowPyrLK(last_frame.frame, cur_frame.frame, // 2 consecutive images 
			                     last_frame.corners, // input point position in first image 
			                     cur_frame.corners, // output point postion in the second image 
			                     status, // tracking success 
		                         err,	// tracking error 
			                     Size(31, 31), 3, criteria, derivlambda, flags);

		//calcOpticalFlowPyrLK(last_frame.frame, cur_frame.frame, last_frame.corners, cur_frame.corners, status, err);
		// 去掉一些不好的特征点
		int k = 0;
		for (size_t i = 0; i<cur_frame.corners.size(); i++)
		{
			if (acceptTrackedPoint(i))
			{
				initial[k] = initial[i];
				last_frame.corners[k] = last_frame.corners[i];
				cur_frame.corners[k++] = cur_frame.corners[i];
			}
		}
		initial.resize(k);
		last_frame.corners.resize(k);
		cur_frame.corners.resize(k);


		//检测跟踪上的哪些点是动态点
		// 4. draw all tracked points
		//cv::cvtColor(im, im, CV_GRAY2RGB);
		Mat resultimg = im.clone();
		Rect brect;
		if (cur_frame.corners.size()>5)
		{
			for (int i = 0; i < cur_frame.corners.size(); i++) {
				// draw circle
				cv::circle(im, cur_frame.corners[i], 1, Scalar(0, 255, 0), 2);
			}
			RotatedRect box = minAreaRect(Mat(cur_frame.corners));//点集的最小外接旋转矩形
			Point2f tr[4], center;
			float radius = 0;
			box.points(tr);
			minEnclosingCircle(Mat(cur_frame.corners), center, radius);//点集的最小外接圆
			for (int i = 0; i < 4; i++)
			{
				line(im, tr[i], tr[(i + 1) % 4], Scalar(0, 255, 0), 3, CV_AA);
			}
			circle(im, center, cvRound(radius), Scalar(0, 255, 255), 3, CV_AA);
			if (!flag_start_track)
			{
				if (230<center.y && 250>center.y && center.x >310 && center.x <330)
				{
					flag_start_track = true;
					brect = box.boundingRect();
					tracker.init(brect, im);
				}
			}
			cout << "tracked keypoints: " << cur_frame.corners.size() << endl;
			imshow("Corner Track", im);
		}
		
		if (flag_start_track)
		{
			result = tracker.update(resultimg);
			rectangle(resultimg, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
			imshow("reslut", resultimg);
		}
		last_frame = cur_frame;
		//if (12 == cur_frame.frame_id)
		//{
		//	
		//	Initial_points cl = detection_initial(last_frame.frame, cur_frame.frame, last_frame.corners, cur_frame.corners);
		//	//Initial_points cl = detection_initial(initial_img, cur_frame.frame, initial, cur_frame.corners);
		//	cur_frame.corners = cl.cur_corners;
		//	initial = cl.initial;
		//	last_frame.corners = cl.initial;
		//}
		//else
		//{
		//	//detection_modle(initial_img, cur_frame.frame, initial, cur_frame.corners, imgsource);
		//	detection_modle(last_frame.frame, cur_frame.frame, last_frame.corners, cur_frame.corners, imgsource);
		//	
		//}
		//cv::cvtColor(im, im, CV_GRAY2RGB);
		////画flow
		//for (size_t i = 0; i<cur_frame.corners.size(); i++)
		//{
		//	circle(im, last_frame.corners[i], 1, Scalar(0, 0, 255), 2);
		//	line(im, last_frame.corners[i], cur_frame.corners[i], Scalar(255, 255, 0));
		//	circle(im, cur_frame.corners[i], 1, Scalar(0, 255, 0), 2);
		//}
	}
			
	
	
}

//决定哪些跟踪点被接受
bool acceptTrackedPoint(int i)
{
	return status[i]&& ((abs(last_frame.corners[i].x - cur_frame.corners[i].x) + abs(last_frame.corners[i].y - cur_frame.corners[i].y)) > 2);;
}


//将跟踪上的点进行FVB算法分析+后面需要添加对极约束
vector<Point2f> detection_modle(Mat initial, Mat cur, vector<Point2f> first, vector<Point2f> next,Mat img)
{
	cout << first.size() << endl;
	cout << next.size() << endl;
	//对跟踪上的点进行描述子的计算，大致分类哪些是动态点，哪些是静态点
	vector<KeyPoint> keypoints_1, keypoints_2;//关键点容器  二维点
	//数据类型进行转换
	KeyPoint::convert(first, keypoints_1, 1, 1, 0, -1);
	KeyPoint::convert(next, keypoints_2, 1, 1, 0, -1);

	Mat descriptors_1, descriptors_2;
	Ptr<DescriptorExtractor> descriptor = ORB::create();//cv3下　ORB描述子
														//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(initial, keypoints_1, descriptors_1);
	descriptor->compute(cur, keypoints_2, descriptors_2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	
	//二进制描述子汉明距离  匹配
	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离 字符串距离  删除 插入 替换次数
	vector<DMatch> matches;//default默认汉明匹配  容器
						   //BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, matches);//对两幅照片的特征描述子进行匹配
	
														  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.
														  //但有时候最小距离会非常小,设置一个经验值30作为下限.
	cout << "total match points: " << matches.size() << endl;
	double min_dist = 0, max_dist = 0;//定义距离

	for (int i = 0; i < descriptors_1.rows; ++i)
	{
		double dist = matches[i].distance;
		if (dist<min_dist) min_dist = dist;
		if (dist>max_dist) max_dist = dist;
	}

	cout << "Max dist :" << max_dist << endl;

	cout << "Min dist :" << min_dist << endl;

	cout << "total match points: " << matches.size() << endl;
	
	std::vector<DMatch> good_matches;
	for (int j = 0; j < descriptors_1.rows; ++j)
	{
		if (matches[j].distance <= max(2 * min_dist, 30.0))
			good_matches.push_back(matches[j]);
	}
	cout << "total good match points: " << good_matches.size() << endl;
	//对匹配完的good_macth数据进行分析

	/*Mat img_match;
	drawMatches(initial, keypoints_1, cur, keypoints_2, good_matches, img_match);
	namedWindow("match", 0);
	imshow("match", img_match);*/
	// 匹配上的点数
	//cout << keypoints_2.size() << endl;


	//计算基础矩阵
	vector<Point2f> points1, points2;

	for (vector<DMatch>::const_iterator it = good_matches.begin();
		it != good_matches.end(); ++it)
	{
		points1.push_back(keypoints_1[it->queryIdx].pt);
		points2.push_back(keypoints_2[it->trainIdx].pt);
	}
	//使用RANSAC算法计算基础矩阵
	//inliers相当于一个掩膜
	vector<uchar> inliers(points1.size(), 0);
	Mat fundamental = findFundamentalMat(
		points1, points2, // matching points
		inliers,         // match status (inlier or outlier)
		FM_RANSAC,
		1.0,      // distance to epipolar line
		0.99     // confidence probability
	);

	//提取合格的匹配项
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM = good_matches.begin();
	vector<DMatch> outMatches;
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM)
	{
		if (*itIn == true)
		{
			outMatches.push_back(*itM);
		}
	}

	//用基础矩阵来矫正匹配点的位置
	vector<Point2f> newPoints1, newPoints2;

	if (IS_REFINE_FUNDA || IS_REFINE_MATCHES)
	{
		//使用RANSAC得出的高质量匹配点再次估算基础矩阵
		points1.clear();
		points2.clear();

		//得到高质量匹配点的坐标
		for (vector<DMatch>::const_iterator it = outMatches.begin();
			it != outMatches.end(); ++it)
		{
			points1.push_back(keypoints_1[it->queryIdx].pt);
			points2.push_back(keypoints_2[it->trainIdx].pt);
		}

		//用八点法计算基础矩阵
		fundamental = findFundamentalMat(
			points1, points2, // matching points
			FM_8POINT); // 8-point method

		if (IS_REFINE_MATCHES)
		{
			correctMatches(fundamental,             // F matrix
				points1, points2,        // original position
				newPoints1, newPoints2); // new position

			for (int i = 0; i< points1.size(); i++)
			{
				keypoints_1[outMatches[i].queryIdx].pt.x = newPoints1[i].x;
				keypoints_1[outMatches[i].queryIdx].pt.y = newPoints1[i].y;

				keypoints_2[outMatches[i].trainIdx].pt.x = newPoints2[i].x;
				keypoints_2[outMatches[i].trainIdx].pt.y = newPoints2[i].y;
			}
		}
	}
	
	KeyPoint::convert(keypoints_2,next);
	int ptCount = (int)outMatches.size();
	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		//pt = keypoints_2[outMatches[i].trainIdx].pt;
		//pt = keypoints_2[i].pt;
		pt = Point2f(newPoints2[i].x, newPoints2[i].y);
		for (vector<Point2f>::iterator it = next.begin(); it != next.end();)
		{
			if (*it == pt)
				it = next.erase(it);
			else
				it++;
		}
	}
	//剩下没匹配上的点个数
	cout << "剩下没匹配上的点个数:" << next.size() << endl;
	if (0 == next.size())
	{
		return next;
	}
	else
	{
		RotatedRect box = minAreaRect(Mat(next));//点集的最小外接旋转矩形
		Point2f tr[4], center;
		float radius = 0;
		box.points(tr);
		minEnclosingCircle(Mat(next), center, radius);//点集的最小外接圆
		cvtColor(img, img, CV_GRAY2BGR);
		for (int i = 0; i < next.size(); i++)
		{
			circle(img, next[i], 3, Scalar(255, 0, 255), CV_FILLED, CV_AA);
		}
		for (int i = 0; i < 4; i++)
		{
			line(img, tr[i], tr[(i + 1) % 4], Scalar(0, 255, 0), 3, CV_AA);
		}
		circle(img, center, cvRound(radius), Scalar(0, 255, 255), 3, CV_AA);
		namedWindow("result", 0);
		imshow("result", img);
		return next;
		//获取到没有匹配上的点，加入对极线约束进行判断
	}
	


	

	//FVB模型

}

vector<Point2f> epipolar_modle(vector<Point2f> points, Mat F)
{
	vector<Point2f> next;//返回值
	const float f11 = F.at<float>(0, 0);
	const float f12 = F.at<float>(0, 1);
	const float f13 = F.at<float>(0, 2);
	const float f21 = F.at<float>(1, 0);
	const float f22 = F.at<float>(1, 1);
	const float f23 = F.at<float>(1, 2);
	const float f31 = F.at<float>(2, 0);
	const float f32 = F.at<float>(2, 1);
	const float f33 = F.at<float>(2, 2);

	
	for (size_t i = 0; i < points.size(); i++)
	{
		const Point2f pt = points[i];
		const float u2 = pt.x;
		const float v2 = pt.y;


	}

	return next;

}






Initial_points detection_initial(Mat initial, Mat cur, vector<Point2f> first, vector<Point2f> next)
{
	Initial_points initial_result;
	//对跟踪上的点进行描述子的计算，大致分类哪些是动态点，哪些是静态点
	vector<KeyPoint> keypoints_1, keypoints_2;//关键点容器  二维点
											  //数据类型进行转换
	KeyPoint::convert(first, keypoints_1, 1, 1, 0, -1);
	KeyPoint::convert(next, keypoints_2, 1, 1, 0, -1);

	Mat descriptors_1, descriptors_2;
	Ptr<DescriptorExtractor> descriptor = ORB::create();//cv3下　ORB描述子
														//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(initial, keypoints_1, descriptors_1);
	descriptor->compute(cur, keypoints_2, descriptors_2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	//二进制描述子汉明距离  匹配
	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离 字符串距离  删除 插入 替换次数
	vector<DMatch> matches;//default默认汉明匹配  容器
						   //BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, matches);//对两幅照片的特征描述子进行匹配

														  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.
														  //但有时候最小距离会非常小,设置一个经验值30作为下限.
	double min_dist = 0, max_dist = 0;//定义距离

	for (int i = 0; i < descriptors_1.rows; ++i)
	{
		double dist = matches[i].distance;
		if (dist<min_dist) min_dist = dist;
		if (dist>max_dist) max_dist = dist;
	}

	std::vector<DMatch> good_matches;
	for (int j = 0; j < descriptors_1.rows; ++j)
	{
		if (matches[j].distance <= max(2 * min_dist, 30.0))
			good_matches.push_back(matches[j]);
	}
	cout << "total good match points: " << good_matches.size() << endl;
	//对匹配完的good_macth数据进行分析 对于第一帧和第二帧之间没匹配上的点进行删除
	//cout << "key points: " << keypoints_1.size() << endl;
	//KeyPoint::convert(keypoints_1, first);
	//KeyPoint::convert(keypoints_2, next);

	//计算基础矩阵
	vector<Point2f> points1, points2;

	for (vector<DMatch>::const_iterator it = good_matches.begin();
		it != good_matches.end(); ++it)
	{
		points1.push_back(keypoints_1[it->queryIdx].pt);
		points2.push_back(keypoints_2[it->trainIdx].pt);
	}
	//使用RANSAC算法计算基础矩阵
	//inliers相当于一个掩膜
	vector<uchar> inliers(points1.size(), 0);
	Mat fundamental = findFundamentalMat(
		points1, points2, // matching points
		inliers,         // match status (inlier or outlier)
		FM_RANSAC,
		1.0,      // distance to epipolar line
		0.99     // confidence probability
	);

	//提取合格的匹配项
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM = good_matches.begin();
	vector<DMatch> outMatches;
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM)
	{
		if (*itIn == true)
		{
			outMatches.push_back(*itM);
		}
	}

	if (IS_REFINE_FUNDA || IS_REFINE_MATCHES)
	{
		//使用RANSAC得出的高质量匹配点再次估算基础矩阵
		points1.clear();
		points2.clear();

		//得到高质量匹配点的坐标
		for (vector<DMatch>::const_iterator it = outMatches.begin();
			it != outMatches.end(); ++it)
		{
			points1.push_back(keypoints_1[it->queryIdx].pt);
			points2.push_back(keypoints_2[it->trainIdx].pt);
		}

		//用八点法计算基础矩阵
		fundamental = findFundamentalMat(
			points1, points2, // matching points
			FM_8POINT); // 8-point method

		
		if (IS_REFINE_MATCHES)
		{
			//用基础矩阵来矫正匹配点的位置
			vector<Point2f> newPoints1, newPoints2;

			correctMatches(fundamental,             // F matrix
				points1, points2,        // original position
				newPoints1, newPoints2); // new position

			for (int i = 0; i< points1.size(); i++)
			{
				keypoints_1[outMatches[i].queryIdx].pt.x = newPoints1[i].x;
				keypoints_1[outMatches[i].queryIdx].pt.y = newPoints1[i].y;

				keypoints_2[outMatches[i].trainIdx].pt.x = newPoints2[i].x;
				keypoints_2[outMatches[i].trainIdx].pt.y = newPoints2[i].y;
			}
			

			initial_result.cur_corners = newPoints1;
			initial_result.initial = newPoints2;
		}
	}
	return initial_result;
}