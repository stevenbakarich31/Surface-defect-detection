#define _USE_MATH_DEFINES
#include <cmath> 


class GMM
{
public:

	int numGaussians; //How many gaussians do we assume?
	double *a; //Mixture proportions
	double *mean;
	double *var;

	//for holding intermediate results
	double *resp;
	double *sum_wj;
	double *sum_wj_xj;
	double *sum_wj_xj2;

	int dataSize;
	double *x; //a pointer to the data of length dataSize

	int currIteration;
	int digits;
	int place;
	int maxIterations; //EM will stop after this many iterations if it hasn't converged
	double precision; //Convergence condition

	bool verbose; //if true, prints information to stderr
	bool iterOnly; //If this and verbose are true, prints only the current iteration to stderr

	double loglikelihood; //of the data given the given model and current parameters
	double BIC; //Bayseian Information Criteria for the data given the given model and current parameters

	void update(); //Update parameters, this folds the E step, M step, loglikelihood, and BIC calculation into a single calculation
	void printState();
	void printIteration();

	double normalLog(double, double, double);

public:

	GMM(int n, double* a_init, double* mean_init, double* var_init, int maxIt, double p, bool v = true, bool iterOnly = false);
	~GMM();

	bool estimate(double* data, int size);

	double getBIC();
	double getLogLikelihood();
	double getMixCoefficient(int i);
	double getMean(int i);
	double getVar(int i);
	

};
//GMM process_gmm(Mat dataPoints);
