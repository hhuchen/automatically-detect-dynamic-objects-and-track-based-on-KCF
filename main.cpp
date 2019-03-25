// get_camera.cpp : �������̨Ӧ�ó������ڵ㡣
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
	std::vector<cv::KeyPoint> cv_feats;  //������
	cv::Mat                   cv_descs;  //����������
	vector<Point2f>           corners;   //KLT������
};


struct Initial_points
{
	vector<Point2f> initial;//��ȡ����KLT������
	vector<Point2f> cur_corners;    // ��ʼ�����ٵ��λ��

};

//variable
std::vector<cv::KeyPoint> cv_feats;  //������
cv::Mat                   cv_descs;  //����������

int frame_id = 0;

Frame last_frame, cur_frame;
Mat initial_img;

vector<Point2f> points[2];  // point0Ϊ�������ԭ��λ�ã�point1Ϊ���������λ��
vector<Point2f> corners;//��ȡ����KLT������
vector<Point2f> initial;    // ��ʼ�����ٵ��λ��

vector<uchar> status;   // ����������״̬��������������Ϊ1������Ϊ0
vector<float> err;

bool flag_start_track = false;//��⶯̬����ɹ����и�������ı�־λ

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
	cout << "��ȡ����������Ϊ��" << corners.size() << endl;
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
	//imshow("ORB������", img_rgb);
	/*cv::Mat im;
	pyrUp(img, im, Size(im.cols * 2, im.rows * 2)); //�Ŵ�һ��
	Ptr<ORB> orb = ORB::create(1500);
	orb->detect(im,cv_feats);
	orb->compute(im,cv_feats,cv_descs);
	cv::Mat img_opencv = cv::Mat(im.size(), CV_8UC3);
	cv::cvtColor(im, img_opencv, CV_GRAY2RGB);
	std::for_each(cv_feats.begin(), cv_feats.end(), [&](cv::KeyPoint i) {
		cv::circle(img_opencv, i.pt, 4 * (i.octave + 1), cv::Scalar(0, 255, 0), 1);
	});
	namedWindow("ORB������", 0);
	imshow("ORB������", img_opencv);*/
}



//����ȡ���Ľǵ���һ����Ԥ����
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


//  ����µ��Ƿ�Ӧ�ñ����
// return: �Ƿ���ӱ�־
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
		
		cout << "��ȡ����������Ϊ��" <<corners.size() << endl;
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

		//���ٳ���
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
		// ȥ��һЩ���õ�������
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


		//�������ϵ���Щ���Ƕ�̬��
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
			RotatedRect box = minAreaRect(Mat(cur_frame.corners));//�㼯����С�����ת����
			Point2f tr[4], center;
			float radius = 0;
			box.points(tr);
			minEnclosingCircle(Mat(cur_frame.corners), center, radius);//�㼯����С���Բ
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
		////��flow
		//for (size_t i = 0; i<cur_frame.corners.size(); i++)
		//{
		//	circle(im, last_frame.corners[i], 1, Scalar(0, 0, 255), 2);
		//	line(im, last_frame.corners[i], cur_frame.corners[i], Scalar(255, 255, 0));
		//	circle(im, cur_frame.corners[i], 1, Scalar(0, 255, 0), 2);
		//}
	}
			
	
	
}

//������Щ���ٵ㱻����
bool acceptTrackedPoint(int i)
{
	return status[i]&& ((abs(last_frame.corners[i].x - cur_frame.corners[i].x) + abs(last_frame.corners[i].y - cur_frame.corners[i].y)) > 2);;
}


//�������ϵĵ����FVB�㷨����+������Ҫ��ӶԼ�Լ��
vector<Point2f> detection_modle(Mat initial, Mat cur, vector<Point2f> first, vector<Point2f> next,Mat img)
{
	cout << first.size() << endl;
	cout << next.size() << endl;
	//�Ը����ϵĵ���������ӵļ��㣬���·�����Щ�Ƕ�̬�㣬��Щ�Ǿ�̬��
	vector<KeyPoint> keypoints_1, keypoints_2;//�ؼ�������  ��ά��
	//�������ͽ���ת��
	KeyPoint::convert(first, keypoints_1, 1, 1, 0, -1);
	KeyPoint::convert(next, keypoints_2, 1, 1, 0, -1);

	Mat descriptors_1, descriptors_2;
	Ptr<DescriptorExtractor> descriptor = ORB::create();//cv3�¡�ORB������
														//-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
	descriptor->compute(initial, keypoints_1, descriptors_1);
	descriptor->compute(cur, keypoints_2, descriptors_2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	
	//�����������Ӻ�������  ƥ��
	//-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ���� �ַ�������  ɾ�� ���� �滻����
	vector<DMatch> matches;//defaultĬ�Ϻ���ƥ��  ����
						   //BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, matches);//��������Ƭ�����������ӽ���ƥ��
	
														  //��������֮��ľ��������������С����ʱ,����Ϊƥ������.
														  //����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
	cout << "total match points: " << matches.size() << endl;
	double min_dist = 0, max_dist = 0;//�������

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
	//��ƥ�����good_macth���ݽ��з���

	/*Mat img_match;
	drawMatches(initial, keypoints_1, cur, keypoints_2, good_matches, img_match);
	namedWindow("match", 0);
	imshow("match", img_match);*/
	// ƥ���ϵĵ���
	//cout << keypoints_2.size() << endl;


	//�����������
	vector<Point2f> points1, points2;

	for (vector<DMatch>::const_iterator it = good_matches.begin();
		it != good_matches.end(); ++it)
	{
		points1.push_back(keypoints_1[it->queryIdx].pt);
		points2.push_back(keypoints_2[it->trainIdx].pt);
	}
	//ʹ��RANSAC�㷨�����������
	//inliers�൱��һ����Ĥ
	vector<uchar> inliers(points1.size(), 0);
	Mat fundamental = findFundamentalMat(
		points1, points2, // matching points
		inliers,         // match status (inlier or outlier)
		FM_RANSAC,
		1.0,      // distance to epipolar line
		0.99     // confidence probability
	);

	//��ȡ�ϸ��ƥ����
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

	//�û�������������ƥ����λ��
	vector<Point2f> newPoints1, newPoints2;

	if (IS_REFINE_FUNDA || IS_REFINE_MATCHES)
	{
		//ʹ��RANSAC�ó��ĸ�����ƥ����ٴι����������
		points1.clear();
		points2.clear();

		//�õ�������ƥ��������
		for (vector<DMatch>::const_iterator it = outMatches.begin();
			it != outMatches.end(); ++it)
		{
			points1.push_back(keypoints_1[it->queryIdx].pt);
			points2.push_back(keypoints_2[it->trainIdx].pt);
		}

		//�ð˵㷨�����������
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
	//ʣ��ûƥ���ϵĵ����
	cout << "ʣ��ûƥ���ϵĵ����:" << next.size() << endl;
	if (0 == next.size())
	{
		return next;
	}
	else
	{
		RotatedRect box = minAreaRect(Mat(next));//�㼯����С�����ת����
		Point2f tr[4], center;
		float radius = 0;
		box.points(tr);
		minEnclosingCircle(Mat(next), center, radius);//�㼯����С���Բ
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
		//��ȡ��û��ƥ���ϵĵ㣬����Լ���Լ�������ж�
	}
	


	

	//FVBģ��

}

vector<Point2f> epipolar_modle(vector<Point2f> points, Mat F)
{
	vector<Point2f> next;//����ֵ
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
	//�Ը����ϵĵ���������ӵļ��㣬���·�����Щ�Ƕ�̬�㣬��Щ�Ǿ�̬��
	vector<KeyPoint> keypoints_1, keypoints_2;//�ؼ�������  ��ά��
											  //�������ͽ���ת��
	KeyPoint::convert(first, keypoints_1, 1, 1, 0, -1);
	KeyPoint::convert(next, keypoints_2, 1, 1, 0, -1);

	Mat descriptors_1, descriptors_2;
	Ptr<DescriptorExtractor> descriptor = ORB::create();//cv3�¡�ORB������
														//-- �ڶ���:���ݽǵ�λ�ü��� BRIEF ������
	descriptor->compute(initial, keypoints_1, descriptors_1);
	descriptor->compute(cur, keypoints_2, descriptors_2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	//�����������Ӻ�������  ƥ��
	//-- ������:������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ�� Hamming ���� �ַ�������  ɾ�� ���� �滻����
	vector<DMatch> matches;//defaultĬ�Ϻ���ƥ��  ����
						   //BFMatcher matcher ( NORM_HAMMING );
	matcher->match(descriptors_1, descriptors_2, matches);//��������Ƭ�����������ӽ���ƥ��

														  //��������֮��ľ��������������С����ʱ,����Ϊƥ������.
														  //����ʱ����С�����ǳ�С,����һ������ֵ30��Ϊ����.
	double min_dist = 0, max_dist = 0;//�������

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
	//��ƥ�����good_macth���ݽ��з��� ���ڵ�һ֡�͵ڶ�֮֡��ûƥ���ϵĵ����ɾ��
	//cout << "key points: " << keypoints_1.size() << endl;
	//KeyPoint::convert(keypoints_1, first);
	//KeyPoint::convert(keypoints_2, next);

	//�����������
	vector<Point2f> points1, points2;

	for (vector<DMatch>::const_iterator it = good_matches.begin();
		it != good_matches.end(); ++it)
	{
		points1.push_back(keypoints_1[it->queryIdx].pt);
		points2.push_back(keypoints_2[it->trainIdx].pt);
	}
	//ʹ��RANSAC�㷨�����������
	//inliers�൱��һ����Ĥ
	vector<uchar> inliers(points1.size(), 0);
	Mat fundamental = findFundamentalMat(
		points1, points2, // matching points
		inliers,         // match status (inlier or outlier)
		FM_RANSAC,
		1.0,      // distance to epipolar line
		0.99     // confidence probability
	);

	//��ȡ�ϸ��ƥ����
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
		//ʹ��RANSAC�ó��ĸ�����ƥ����ٴι����������
		points1.clear();
		points2.clear();

		//�õ�������ƥ��������
		for (vector<DMatch>::const_iterator it = outMatches.begin();
			it != outMatches.end(); ++it)
		{
			points1.push_back(keypoints_1[it->queryIdx].pt);
			points2.push_back(keypoints_2[it->trainIdx].pt);
		}

		//�ð˵㷨�����������
		fundamental = findFundamentalMat(
			points1, points2, // matching points
			FM_8POINT); // 8-point method

		
		if (IS_REFINE_MATCHES)
		{
			//�û�������������ƥ����λ��
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