
// Steel_Defect_Detection_Classification


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp> 
#include <opencv2/imgproc.hpp> 
#include "opencv2/imgcodecs.hpp" 
#include "opencv2/features2d/features2d.hpp" 
#include <opencv2/highgui.hpp> 
#include <opencv2/ml.hpp> 
#include <iostream>
#include "WienerFilter.h"



using namespace std;
using namespace cv;
using namespace cv::ml;


Mat Load_img();
Mat Preprocess_img(Mat);
void insertionSort(int[], int);
vector<Mat> Defect_Detection(Mat);
vector<Mat> Divide_Image(Mat, int, int);
array<double, 3> Featuers_Extraction(Mat, int, int);
double Discriminant_function(Mat, array<double, 3>);
vector<Mat> Multivariate_Normal_Distribution(Mat);
//Mat Defect_Classification(Mat, Mat);


/*****************************************************************
***************************** Main *******************************
******************************************************************/
int main()
{
	Mat Steel_Image;  // input image
	Mat PreImg;   // preprocessing image
	vector<Mat> Def_Blocks;   // defected blocks


	/////////////////////// Load Image ///////////////////////
	Steel_Image = Load_img();  // load image from database


	////////////////// PreProcessing Image ///////////////////
	PreImg = Preprocess_img(Steel_Image);   // return preprocessing image


	////////////////// Defect Detection //////////////////////
	Def_Blocks = Defect_Detection(Steel_Image);   // return defected blocks


	//////////////// Defect Classification ///////////////////
	//Defect_Classification(PreImg, Def_Blocks);     // display defect type

	return 0;
}



/******************************************************************
******************** Load Image Function ***********************
*******************************************************************/
Mat Load_img()
{
	Mat Steel_Image;

	Steel_Image = imread("Pa_1.bmp", CV_LOAD_IMAGE_GRAYSCALE);    // read image

	if (!Steel_Image.data)     // check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		//return -1;
	}

	namedWindow("Load Image", WINDOW_NORMAL);
	imshow("Steel_Image", Steel_Image);
	waitKey();

	return Steel_Image;
}



/******************************************************************
******************** Preprocessing Function ***********************
*******************************************************************/
Mat Preprocess_img(Mat Steel_Image)
{
	int M, N;  // size image
	Mat Steel_Median_filter;
	Mat Steel_Wiener_filter;

	M = 200;
	N = 200;

	//////////////// Resize the image to M x N //////?/////////

	int rows = Steel_Image.rows;  // image width
	int columns = Steel_Image.cols;  // image height


	if (rows != M || columns != N) // Check for image size
	{
		cv::resize(Steel_Image, Steel_Image, cv::Size(), M, N); // Resize Image to M،؟N
	}


	///////////////////// Median Filter ///////////////////////
	int window[9];

	Steel_Median_filter = Steel_Image.clone();

	for (int y = 1; y < Steel_Image.rows - 1; y++) {
		for (int x = 1; x < Steel_Image.cols - 1; x++) {

			// Pick up window element
			window[0] = Steel_Image.at<uchar>(y - 1, x - 1);
			window[1] = Steel_Image.at<uchar>(y, x - 1);
			window[2] = Steel_Image.at<uchar>(y + 1, x - 1);
			window[3] = Steel_Image.at<uchar>(y - 1, x);
			window[4] = Steel_Image.at<uchar>(y, x);
			window[5] = Steel_Image.at<uchar>(y + 1, x);
			window[6] = Steel_Image.at<uchar>(y - 1, x + 1);
			window[7] = Steel_Image.at<uchar>(y, x + 1);
			window[8] = Steel_Image.at<uchar>(y + 1, x + 1);

			// sort the window to find median
			insertionSort(window, 9);

			// assign the median to centered element of the matrix
			Steel_Median_filter.at<uchar>(y, x) = window[4];
		}
	}
	cout << Steel_Median_filter.rows << Steel_Median_filter.cols;
	namedWindow("Image After Median Filter", WINDOW_NORMAL);
	imshow("Image After Median Filter", Steel_Median_filter);
	waitKey(0);

	///////////////////// Wiener Filter //////////////////////

	double estimatedNoiseVariance;
	Steel_Wiener_filter = Steel_Median_filter.clone();

	// Call to WienerFilter function with a 3x3 kernel and estimated noise variances
	estimatedNoiseVariance = WienerFilter(Steel_Median_filter, Steel_Wiener_filter, Size(3, 3));
	namedWindow("Image After Median Filter and Wiener Filter", WINDOW_NORMAL);
	imshow("Image After Median Filter and Wiener Filter", Steel_Wiener_filter);
	waitKey(0);

	return Steel_Wiener_filter;
}

void insertionSort(int window[], int n)
{
	int i, key, j;
	for (i = 1; i < n; i++)
	{
		key = window[i];
		j = i - 1;

		/* Move elements of arr[0..i-1], that are greater
		than key, to one position ahead of their current position */
		while (j >= 0 && window[j] > key)
		{
			window[j + 1] = window[j];
			j = j - 1;
		}
		window[j + 1] = key;
	}
}



/*****************************************************************
******************* Defect Detection Function ********************
******************************************************************/
vector<Mat> Defect_Detection(Mat PreImg)
{
	vector<Mat> samples;  // two groups
	int num_blocks;  // number of blocks
	double resulte;
	Mat Defected_Blocks;  // array contains defected blocks
	int num_Defected_Blocks;  // number of defected blocks
	Mat block;




	int M = 200;
	int N = 200;

	int W = 40;
	int H = 40;

	
	// get two groups from Multivariate Normal Distribution by Maximum Likelihood
	samples = Multivariate_Normal_Distribution(PreImg);


	// defected group and non-defected group
	// G1 = samples[1]; // defected group
	// G2 = samples[2]; // not defected group

	// calculate statistical features for the groups
	//MV1 = Featuers_Extraction(G1);  // statistical features for group 1
	//MV2 = Featuers_Extraction(G2);  // statistical features for group 2

	// calculate common covariance


	vector<Mat>  Steel_Image_Blocks;  // array contains image blocks

									  // Divide the image into W x H size blocks
	Steel_Image_Blocks = Divide_Image(PreImg, M, N);  // return image blocks

	num_blocks = 25;
	//num_blocks = Steel_Image_Blocks.size();   // number of blocks

	num_Defected_Blocks = 0;  // initialization value
	array<double, 3>  block_featuers;  // array contains mean vetor of the current block




	for (int i = 0; i < num_blocks; i++)
	{
		block = Steel_Image_Blocks[i];  // one block from image
										//	imshow("block number" + i, block);
										//	waitKey(0);

										// calculate statistical featuers for one block
		block_featuers = Featuers_Extraction(Steel_Image_Blocks[i], W, H);

		// calculate discrimininate function
		//disc = ((MV1 - MV2)' * inv(CCV) ) * MVBlock -  (0.5 * (MV1 - MV2)' *  inv(CCV) * (MV1 - MV2))


		resulte = 0;
		if (resulte = 0)
		{
			//show defected block
			num_Defected_Blocks = num_Defected_Blocks + 1;
			//Defected_Blocks[num_Defected_Blocks] = block;
		}
	}

	if (num_Defected_Blocks == 0)
	{
		cout << "the image is not defected" << std::endl;
	}
	else
	{
		cout << "the image is defected" << std::endl;
	}

	return Steel_Image_Blocks;
}



/******************************************************************
**************** Multivariate_Normal_Distribution *****************
*******************************************************************/
vector<Mat> Multivariate_Normal_Distribution(Mat sss)
{
	cv::Mat dataPoints = cv::imread("Pa_1.bmp");

	//convert the preprocessing image to float image
	cv::Mat floatSource;
	dataPoints.convertTo(floatSource, CV_32F);

	///////////// form the training samples //////////
	cv::Mat samples(dataPoints.rows * dataPoints.cols, 3, CV_32FC1);   //convert the float image to column vector

	int idx = 0;
	for (int y = 0; y < dataPoints.rows; y++)
	{
		cv::Vec3f* row = floatSource.ptr<cv::Vec3f >(y);
		for (int x = 0; x < dataPoints.cols; x++)
		{
			samples.at<cv::Vec3f >(idx++, 0) = row[x];
		}
	}


	//////////// cluster the data using EM algorithm ////////////
	cout << "Starting EM training" << endl;
	Mat labels;
	Ptr<EM> em_model = EM::create();
	em_model->setClustersNumber(2);   //we need just 2 clusters
	em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
	em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 0.1));
	em_model->trainEM(samples, noArray(), labels, noArray());
	cout << "Finished training EM" << endl;


	Mat means = em_model->getMeans();       //the two dominating colors
	Mat weights = em_model->getWeights();   //the weights of the two dominant colors

	const int fgId = weights.at<double>(0) > weights.at<double>(1) ? 0 : 1;



	////////// classify each of the block pixels ///////////
	cv::Mat fgImg(dataPoints.rows, dataPoints.cols, CV_8UC3);
	cv::Mat bgImg(dataPoints.rows, dataPoints.cols, CV_8UC3);
	
	vector<Mat> groups;
	idx = 0;
	for (int y = 0; y < dataPoints.rows; y++)
	{
		for (int x = 0; x < dataPoints.cols; x++)
		{
			//classify
			const int result = cvRound(em_model->predict(samples.row(idx++)));

			//set either foreground or background
			if (result == fgId)
			{
				fgImg.at<cv::Point3_<uchar> >(y, x) = dataPoints.at<cv::Point3_<uchar> >(y, x);
			}
			else
			{
				bgImg.at<cv::Point3_<uchar> >(y, x) = dataPoints.at<cv::Point3_<uchar> >(y, x);
			}

		}
	}

	// 
	normalize(fgImg, fgImg, 0, 255, NORM_MINMAX);
	normalize(bgImg, bgImg, 0, 255, NORM_MINMAX);


	cv::imshow("Foreground", fgImg);
	cv::imshow("Background", bgImg);



	char key = (char)waitKey();


	return groups;
}


/********************************************************************
*********************** Divide Image to Blocks **********************
*********************************************************************/
vector<Mat> Divide_Image(Mat PreImg, int rows, int columns)
{
	int blockSizeR = 40;
	int blockSizeC = 40;
	int number_block;

	number_block = (rows / blockSizeR) * (columns / blockSizeC);

	cout << "Number rows of the image     " << rows << "    \n"
		<< "Number columns of the image    " << columns << "    \n "
		<< "Number of Blocks theat image is divided     " << number_block << "    \n"
		<< "Number rows of the blocks    " << blockSizeR << "      \n"
		<< "Number columns of the blocks    " << blockSizeC << "\n";


	vector<Mat> Steel_image_divide;
	int k = 0, m = 0;
	int h = (int)(number_block / 5);

	for (int i = 0; i < number_block; i++)
	{
		if (k == h)
		{
			k = 0;
			if (i >= 5)
				m = 1;
			if (i >= 10)
				m = 2;
			if (i >= 15)
				m = 3;
			if (i >= 20)
				m = 4;
		}

		Steel_image_divide.push_back(Mat(PreImg, cv::Rect(0 + (k * 40), 0 + (m * 40), 40, 40)).clone());
		k++;
	}

	Mat im1((Steel_image_divide[0].rows * 5) + 1000, (Steel_image_divide[0].cols * 5) + 1000, CV_8U);

	k = 0, m = 0;

	for (int i = 0; i < number_block; i++) {
		if (k == h)
		{
			k = 0;
			if (i >= 5)
				m = 1;
			if (i >= 10)
				m = 2;
			if (i >= 15)
				m = 3;
			if (i >= 20)
				m = 4;
		}
		Steel_image_divide[i].copyTo(im1(Rect(0 + (k*Steel_image_divide[i].cols) + (10 * k),
			0 + (m*Steel_image_divide[i].rows) + (20 * m),
			Steel_image_divide[i].cols, Steel_image_divide[i].rows)));
		k++;
	}
	namedWindow("Blocks After Preprocessing", WINDOW_NORMAL);
	imshow("Blocks After Preprocessing", im1);
	waitKey(0);

	return Steel_image_divide;
}



/*****************************************************************
*********** Extract Statistical Feature of the Blocks ************
******************************************************************/
array<double, 3> Featuers_Extraction(Mat block, int rows, int cols)
{
	double Difference, Max, Min, sum = 0.0, Mean, sum1 = 0.0, Var, pw;
	array<double, 3> feature_Block;
	vector<std::array<double, 3>> vec;



	// identify the maximum and minimum value on the block i to get the Difference
	minMaxLoc(block, &Min, &Max);
	Difference = Max - Min;
	feature_Block[0] = Difference;

	// find the Mean
	for (int xx = 0; xx <rows; xx++)
	{
		for (int y = 0; y < cols; y++)
		{
			sum = sum + (double)block.at<uchar>(xx, y);
		}
	}
	Mean = sum / (rows*cols);
	feature_Block[1] = Mean;

	//find The Variance
	for (int xx = 0; xx < rows; xx++)
	{
		for (int y = 0; y < cols; y++)
		{
			pw = pow(((double)block.at<uchar>(xx, y) - Mean), 2);
			sum1 = sum1 + pw;
		}
	}
	Var = sum1 / (rows * cols);
	feature_Block[2] = Var;
	vec.push_back(feature_Block);

	/////////// Display the Statistical Feature of the Blocks //////////
	for (int i = 0; i < 3; i++)
	{
		cout << feature_Block[i] << "   ";
	}
	cout << endl;
	waitKey(0);


	return feature_Block;
}


/******************************************************************
********************** Discriminant Function **********************
*******************************************************************/
double Discriminant_function(Mat block, array<double, 3> MVblock)
{
	double disc;
	vector<Mat> MV1, MV2;
	vector<Mat> samples;



	// G1 = samples[1]; // defected group
	// G2 = samples[2]; // not defected group

	//MV1 = Featuers_Extraction(G1);  // statistical features for group 1
	//MV2 = Featuers_Extraction(G2);  // statistical features for group 2


	// calculate covarianve for group 1
	//sum(:, : ) = 0;
	//for n = 1 : N1
	//sum = sum(:, : ) + ((G1(n, 1) - MV1)*(G1(n, 1) - MV1)');
	//end
	//CV1 = (1\(N1 - 1))* sum;

	// calculate covarianve for group 2
	//sum(:, : ) = 0;
	//for n = 1 : N2
	//sum = sum(:, : ) + ((G2(n, 1) - MV2)*(G2(n, 1) - MV2)');
	//end
	//CV2 = (1\(N2 - 1))* sum;

	// common covariance
	//CCV = 0.5 * (CV1 + CV2);=

	// calculate discrimininate function
	//disc = ((MV1 - MV2)' * inv(CCV) ) * MVBlock -  (0.5 * (MV1 - MV2)' *  inv(CCV) * (MV1 - MV2))

	disc = 0;
	return disc;
}

// ended


