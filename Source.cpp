/***************************************************************************/
/*************** Steel Defect Detection Classification Project *************/
/***************************************************************************/

#define _USE_MATH_DEFINES

#include <cmath>  
#include <iomanip>
#include <limits>
#include <ctime>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>  
#include <opencv2/imgproc.hpp> 
#include "opencv2/imgcodecs.hpp" 
#include "opencv2/features2d/features2d.hpp" 
#include <opencv2/highgui.hpp> 
#include <opencv2/ml.hpp> 

#include "WienerFilter.h"
#include "MND.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

Mat Load_img();
Mat Preprocess_img(Mat);
void insertionSort(int[], int);
vector<Mat> Defect_Detection(Mat);
vector<Mat> Multivariate_Normal_Distribution(Mat);
Mat Featuers_Extraction(Mat);
Mat Covariance(Mat, Mat);
vector<Mat> Divide_Image(Mat, int, int);
//Mat Defect_Classification(Mat, Mat);


/***************************************************************************/
/*                               Main                                      */
/***************************************************************************/
int main()
{
	Mat Steel_Image;  // input image
	Mat PreImg;   // preprocessing image
	vector<Mat> Def_Blocks;   // defected blocks


    //////////////////////// Load Image ////////////////////////
	Steel_Image = Load_img();   // load image from database

	
	//////////////////// PreProcessing Image ///////////////////
	PreImg = Preprocess_img(Steel_Image);   // return preprocessing image


	//////////////////// Defect Detection //////////////////////
	Def_Blocks = Defect_Detection(PreImg);   // return defected blocks


	//////////////// Defect Classification /////////////////////
	//Defect_Classification(PreImg, Def_Blocks);    // display defect type

	waitKey(0);

	return 0;
}


/***************************************************************************/
/*                          Load Image Function                            */
/***************************************************************************/
Mat Load_img()
{
	Mat Steel_Image;

	// read the image from NEU database
	Steel_Image = imread("Pa_1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	// check for invalid input															  
	if (!Steel_Image.data)
	{
		cout << "Could not open or find the image" << "\n" ;
		//return -1;
	}

	// display the image
	//imshow("Steel Image from database", Steel_Image);
	 
	return Steel_Image;
}



/***************************************************************************/
/*                           Preprocessing Function                        */
/***************************************************************************/
Mat Preprocess_img(Mat Steel_Image)
{
	int M, N;
	Mat Steel_Median_filter;
	Mat Steel_Wiener_filter;

	M = 200;
	N = 200;

	//////////////// Resize the image to M x N ////////////////

	int rows = Steel_Image.rows;  // image width
	int columns = Steel_Image.cols;  // image height


	if (rows != M || columns != N) // Check for image size
	{
		// Resize Image to M × N
		cv::resize(Steel_Image, Steel_Image, cv::Size(), M, N);
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

	// display image
	//imshow("Image After Median Filter", Steel_Median_filter);

	///////////////////// Wiener Filter //////////////////////

	double estimatedNoiseVariance;
	Steel_Wiener_filter = Steel_Median_filter.clone();

	// Call to WienerFilter function with a 3x3 kernel and estimated noise variances
	estimatedNoiseVariance = WienerFilter(Steel_Median_filter, Steel_Wiener_filter, Size(3, 3));

	// display image
	//imshow("Image After Median Filter and Wiener Filter", Steel_Wiener_filter);


	cout << "Number rows of the image       " << rows    << "\n"
		 << "Number columns of the image    " << columns << "\n" << "\n";

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



/***************************************************************************/
/*                       Defect Detection Function                         */
/***************************************************************************/
vector<Mat> Defect_Detection(Mat PreImg)
{
	int M = PreImg.rows;
	int N = PreImg.cols;


	///////////// Multivariate Normal Distribution ////////////
	
	vector<Mat> samples; // vector contains two groups of pixels
	samples = Multivariate_Normal_Distribution(PreImg);

	// get two groups from Multivariate Normal Distribution by Maximum Likelihood
	Mat G1, G2;
	samples[0].copyTo(G1);   // defected group
	samples[1].copyTo(G2);   // non-defected group

   ///// calculate statistical featuers for two groups ///////
/*
	Mat  MV1, MV2;  // array contains mean vetor for image
	
	cout << "The Statistical Features for group1 (defected group) are" << "\n";
	MV1 = Featuers_Extraction(G1);  // statistical features for group 1

	cout << "The Statistical Features for group 2 (non-defected group) are" << "\n";
	MV2 = Featuers_Extraction(G2);  // statistical features for group 2

	/////////////// calculate common covariance //////////////
									
	Mat CV1, CV2;

	// calculate covariance for group1 and group2 
	CV1 = Covariance(G1, MV1); // for group 1
	CV2 = Covariance(G2, MV2); // for group 2

	// common covariance
	Mat SUM_CV, CCV;

	// CCV = 0.5 * (CV1 + CV2)
	add(CV2, CV1, SUM_CV); 
	multiply(0.5, SUM_CV, CCV);  
	
*/
	///////// Divide the image to W x H size blocks //////////
	
	int W = 40;
	int H = 40;
	vector<Mat>  Steel_Image_Blocks;  // array contains image blocks

	Steel_Image_Blocks = Divide_Image(PreImg, M, N);  // image blocks

	int num_blocks = (int) Steel_Image_Blocks.size(); // number of blocks

	///////////////// Discriminant function ////////////////
	/*
	calculate discriminant function for each block to decide if it is defected or not
	if the value of discriminant positev then discriminant then the block is defected.
	discriminant function = transpose(MV1 - MV2) * inv(CCV) - 0.5 * transpose(MV1 - MV2) * inv(CCV) * (MV1 + MV2)
	*/

	vector<Mat> Defected_Blocks;  // array contains defected blocks
	int num_Defected_Blocks = 0;  // number of defected blocks
	Mat defected_image; // image contain only defected blocks
/*
	//calculate (MV1 - MV2)
	Mat  SUB_MV;
	subtract(MV1, MV2, SUB_MV); 

	// calculate transpot(MV1 - MV2)
	Mat SUB_MV_tp = cv::Mat(3, 1, CV_32F);
	transpose(SUB_MV, SUB_MV_tp);

	// calculate inverse(CCV)
	Mat CCV_inv(CCV.rows, CCV.cols, CV_32F);
	CCV_inv = CCV.inv();
	invert(CCV, CCV_inv);

	// calculate (MV1 + MV2)
	Mat ADD_MV;
	add(MV1, MV2, ADD_MV);

	// calculate 0.5 * transpose(MV1 - MV2) * inv(CCV) * (MV1 + MV2)
	Mat p1, p2;
	multiply(-0.5, SUB_MV_tp, p1);
	multiply(p1, CCV_inv, p1);
	multiply(p1, ADD_MV, p2);
*/
	// calculate discriminant for each blocks
	for (int i = 0; i < num_blocks; i++)
	{
		Mat block = Steel_Image_Blocks[i];  // one block from image
	
	   imshow("block" + i , block);

	   // Extract statistical featuers for each Block
	   //array<double, 3>  MVBlock;  // array contains mean vetor of the current block
/*
	   cout << "The Statistical Features for block  " << i+1 << "  is:" << "\n";
	   MVBlock = Featuers_Extraction(Steel_Image_Blocks[i], W, H);  // calculate statistical featuers for one bloc

	   // calculate transpose(MVBlock) 
	   Mat MVBlock_tp = cv::Mat(3, 1, CV_32F);
	   transpose(MVBlock, MVBlock_tp);


	   // calculate transpose(MV1 - MV2) * inv(CCV) * transpose(MVBlock)
	   Mat p3;
	   multiply(p1, MVBlock_tp, p3);

	   // discriminant function = transpose(MV1 - MV2) * inv(CCV) - 0.5 * transpose(MV1 - MV2) * inv(CCV) * (MV1 + MV2)
	
	  // Mat disc;
	  //subtract(p3, p2, disc);

	   if (disc = 0)
	   {
		   imshow("block number" + i, block);
	       
		   // add border to defected block
		   GaussianBlur(block, block, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
	       num_Defected_Blocks++ ;
	       
		   // add defected block to defected blocks matrix
		   Defected_Blocks.push_back = block;
	     
		   // add defected block to defected image 

	
	    } */
	}

	/////// calssify the image to defected or non-defected image ///////
	if (num_Defected_Blocks == 0)
	{
		cout << "the image is non-defected" << "\n";
	}
	else
	{
		cout << "the image is defected" << "\n"
	    	 << " number of defected blocks is  " << num_Defected_Blocks << endl;
		//display defected image
	}


	///////////////////////// ended /////////////////////////
	cout << "Defected Detection Step Ended";
	waitKey(0);

	//Mat Defected_Blocks;
	return defected_image;
}

 

/***************************************************************************/
/*                    Multivariate_Normal_Distribution                     */
/***************************************************************************/
vector<Mat> Multivariate_Normal_Distribution(Mat Points)
{
	Mat dataPoints = imread("Pa_1.bmp");
	
	// Check for invalid input
	if (dataPoints.empty())
	{
		cout << "Could not open or find the image" << std::endl;
	}

	int numPoints = dataPoints.rows*dataPoints.cols;
	cout << "number of pixels in image	" << numPoints << endl;

	const int gaussians = 2;
	const size_t maxIterations = 250;
	const double tolerance = 1e-10;
	
	double *W = new double[2];
	W[0] = 0.8;
	W[1] = 0.2;

	double *Mu = new double[2];
	Mu[0] = 100;
	Mu[1] = 200;

	double *Sigma = new double[2];
	Sigma[0] = 15;
	Sigma[1] = 10;

	
    // ????????????????????????
	GMM gmm(gaussians, W, Mu, Sigma, maxIterations, tolerance, true, true);

	//convert the preprocessing image to float image
	Mat floatSource;
	dataPoints.convertTo(floatSource, CV_32F);
	
	/////////////// form the training samples //////////////
	double *data = new double[numPoints];
	int idx = 0;  // counter
	
	for (int y = 0; y < dataPoints.rows; y++)
	{
		Vec3f* row = floatSource.ptr<Vec3f>(y);
	    for (int x = 0; x < dataPoints.cols; x++)
		{
			data[idx++] = row[x][0];
	    }
	}
	
	// 
	cout << "Starting EM training" << endl;
	gmm.estimate(data, numPoints);
	cout << endl << "Finished training EM" << endl << endl;
	
	////////////// Classification of pixels ////////////// 
	
	Mat G1, G2;  // G1 = defected group, G2 = non-defected group
	int num1 = 0, num2 = 0; // number of pixels in each group

	
	//check for all pixels if it is belonge to defected group or non defected group
	for (int i = 0; i < numPoints; i++)
	{
		double x = data[i];

		///// calculate Maximum Likelihood for normal distribuation
	    double L1 = gmm.normalLog(x, gmm.mean[0], gmm.var[0]);
	    double L2 = gmm.normalLog(x, gmm.mean[1], gmm.var[1]);
	    if (L1 > L2) 
		{
			num1++;
			G1.push_back(x);
		}
		else if (L1 < L2)
		{
			num2++;
			G2.push_back(x);
		}
		else
		{
			int r = rand();
			if (r > 50)
			{
				num2++;
				G1.push_back(x);
			}
			else
			{
				num1++;
				G2.push_back(x);
			}
		}
	}

	cout << "number of pixel in defected group	" << num1 << "\n";
	cout << "number of pixel in non-defected group	" << num2 << "\n" << "\n";

	// display Maximum likelihood rule for normal distributions
	Mat image = Mat::zeros(200, 300, CV_8UC3);
	image.setTo(0);
	cv::line(image, cv::Point(0, 190), cv::Point(300, 190), Scalar(255, 255, 255), 1);
	cv::line(image, cv::Point(10, 0), cv::Point(10, 200), Scalar(255, 255, 255), 1);
	
	// display Maximum likelihood 1
	for (int i = 0; i < 300 - 11; i++)
	{
		int L1 = (int)(exp(gmm.normalLog(i, gmm.mean[0], gmm.var[0])) * 5000);
		int L2 = (int)(exp(gmm.normalLog(i + 1, gmm.mean[0], gmm.var[0])) * 5000);
	    cv::line(image, cv::Point(i + 10, 190 - L1), cv::Point(i + 11, 190 - L2), Scalar(255, 255, 0), 1);
	}
	
	// display Maximum likelihood 2
	for (int i = 0; i < 300 - 11; i++)
	{
		int L1 = (int)(exp(gmm.normalLog(i, gmm.mean[1], gmm.var[1])) * 5000);
	    int L2 = (int)(exp(gmm.normalLog(i + 1, gmm.mean[1], gmm.var[1])) * 5000);
	    cv::line(image, cv::Point(i + 10, 190 - L1), cv::Point(i + 11, 190 - L2), Scalar(255, 255, 0), 1);
	}

    // display normal distributions image
  	imshow("Maximum likelihood rule for normal distributions", image);

	cout << "W = " << gmm.a[0] << " " << gmm.a[1] << endl;
	cout << "W = " << gmm.a[0] << " " << gmm.a[1] << endl;
	cout << "Mu = " << gmm.mean[0] << " " << gmm.mean[1] << endl;
	cout << "Sigma = " << gmm.var[0] << " " << gmm.var[1] << endl;
	cout << "Maximum Likelihood=" << gmm.loglikelihood << endl;
	cout << "BIC=" << gmm.BIC << endl;

	

	// put group1 and group2 in one vector
	vector<Mat> groups;
	groups.push_back(G1);
	groups.push_back(G2);

	return groups;
}




/*****************************************************************/
/*                  Extract Statistical Feature                  */
/*****************************************************************/
Mat Featuers_Extraction(Mat block)
{
	int rows = block.rows; // number of rows in block
	int cols = block.cols; // number of cloumns in block
	Mat feature_Block; // mean, variance and difference

	////////////////// find the difference //////////////////
	double Max = 0;
	double Min = 255;
	/*
	// identify the maximum and minimum value to get the Difference
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			uchar val = block.at< uchar >(i, j);
			if (val > Max) Max = val;
			if (val < Min) Min = val;
		} 
	}
	*/
	// calculate the difference
	double Difference = Max - Min;
	feature_Block.push_back(Difference);


	/////////////////// find the mean ///////////////////////

	double sum = 0.0, mean;
	/*
	for (int y = 1; y < rows; y++)
	{
	for (int x = 0; x < cols; x++)
	{
	sum += (double)block.at<uchar>(0, 0);
	}
	}
	mean = sum /  (rows * cols);;
	*/
	
	feature_Block.push_back(mean);


	//////////////////////// find the variance ///////////////////
	
	double var, pow;
	sum = 0.0;
	/*
	for (int x = 0; x < rows; y++)
	{
		for (int y = 0; y < cols; x++)
		{
			pow = pow(((double)block.at<uchar>(x, y) - mean), 2);
	        sum = sum + pow;
	}
	}
	var = sum / (rows * cols);
	feature_Block[2] = var;
	*/

	var = 255;
	feature_Block.push_back(var);


	////// Display the Statistical Features of the Blocks //////////
	/* not work
	cout << "Mean:        " << feature_Block.at<double>(0, 0) << "\n"
		<< "Difference:  " << feature_Block.at<double>(0, 1) << "\n"
		<< "Variance:    " << feature_Block.at<double>(0, 2) << "\n" << "\n";
     */

	return feature_Block;
}


/******************************************************************/
/*                     Covariance Function                        */
/******************************************************************/
Mat Covariance(Mat group, Mat MV)
{
	Mat cov;  // covariance
	Mat sum;

	int num = group.rows; // number of pixels in group 1


	for (int y = 0; y < num; y++)
	{
		// computes
		//double x = (double)group.at<uchar>(0, 0);

		Mat data, data_tp;

		// calculate (x - MV)
		//subtract(x, MV, data);

		// calculate transpose (x - MV)
		//transpose(data, data_tp); // 

		//Mat xx(3, 3, CV_32F);

		// calculate (x - MV) * transpose(x - MV)
		//multiply(data, data_tp, xx);

		// sum  
		//add(xx, sum, sum);
	}

	// calculate 
	//double n = 1 / (num - 1);
	//multiply(n, sum, cov);

	return cov;
}



/********************************************************************/
/*                        Divide Image to Blocks                    */
/********************************************************************/
vector<Mat> Divide_Image(Mat PreImg, int rows, int columns)
{
	int blockSizeR = 40;
	int blockSizeC = 40;

	int numBlockR = rows / blockSizeR;
	int numBlockC = columns / blockSizeC;

	int number_block = numBlockR * numBlockR;

	cout << "Number of Blocks that image is divided:     " << number_block << "    \n" << "\n"
		<< "Number of rows  in each blocks:    " << blockSizeR << "      \n"
		<< "Number of columns in blocks:  " << blockSizeC << "\n";

	vector<Mat> Steel_image_divide;
	int k = 0, m = 0;
	int h = (int)(number_block / numBlockR);


	for (int i = 0; i < number_block; i++)
	{
		if (k == h)
		{
			k = 0;
			if (i >= 5)  m = 1;
			if (i >= 10) m = 2;
			if (i >= 15) m = 3;
			if (i >= 20) m = 4;
		}

		Steel_image_divide.push_back(Mat(PreImg, Rect((k * 40), (m * 40), 40, 40)).clone());
		k++;
	}

	Mat im1((Steel_image_divide[0].rows * numBlockR) + 50, (Steel_image_divide[0].cols * numBlockC) + 50, CV_8U);

	k = 0, m = 0;

	for (int i = 0; i < number_block; i++)
	{
		if (k == h)
		{
			k = 0;
			if (i >= 5)  m = 1;
			if (i >= 10) m = 2;
			if (i >= 15) m = 3;
			if (i >= 20) m = 4;
		}

		Steel_image_divide[i].copyTo(im1(Rect((k*Steel_image_divide[i].cols) + (10 * k),
			                                  (m*Steel_image_divide[i].rows) + (10 * m),
			                                   Steel_image_divide[i].cols, Steel_image_divide[i].rows)));
		k++;
	}

	// display the divided image
	//imshow("Blocks After Preprocessing", im1);

	return Steel_image_divide;
}



/***************************************************************************/
/***************************** Ended Project *******************************/
/***************************************************************************/
   