/********************************************
*  mcxyzCUDA.c,	in ANSI Standard C programing language with NVIDIA extensions for CUDA
*      Usage:  mcxyzCUDA myname and myname_T.bin
*      which loads myname_H.mci, and saves myname_F.bin.
*
*	created 2010, 2012 by
*	Steven L. JACQUES
*  Ting LI
*	Oregon Health & Science University
*   CUDA version by
*	Alexander Doronin (https://github.com/aledoronin)
*
*
*  USAGE   mcxyz myname
*              where myname is the user's choice.
*          The program reads two files prepared by user:
*                  myname_H.mci    = header input file for mcxyz
*                  myname_T.bin    = tissue structure file
*          The output will be written to 3 files:
*                  myname_OP.m     = optical properties  (mua, mus, g for each tissue type)
*                  myname_F.bin    = fluence rate output F[i] [W/cm^2 per W delivered]
*
*  The MATLAB program maketissue.m can create the two input files (myname_H.mci, myname_T.bin).
*
*  The MATLAB program lookmcxyz.m can read the output files and display
*          1. Fluence rate F [W/cm^2 per W delivered]
*          2. Deposition rate A [W/cm^3 per W delivered].
*
*  Log:
*  Written by Ting based on Steve's mcsub.c., 2010.
*      Use Ting's FindVoxelFace().
*	Use Steve's FindVoxelFace(), Dec. 30, 2010.
*  Reorganized by Steve. May 8, 2012:
*      Reads input files, outputs binary files.
**********/

#include <time.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

#define GTX_980_TI

#ifdef GTX_980_TI

#define BLOCK_DIM_MCXYZ 88
#define THREADS_DIM_MCXYZ 512
#define PHOTONS_MCXYZ (BLOCK_DIM_MCXYZ * THREADS_DIM_MCXYZ)

#endif

#define Ntiss		19          /* Number of tissue types. */
#define STRLEN 		32          /* String length. */
#define ls          1.0E-7      /* Moving photon a little bit off the voxel face */
#define	PI          3.1415926
#define	LIGHTSPEED	2.997925E10 /* in vacuo speed of light [cm/s] */
#define ALIVE       1   		/* if photon not yet terminated */
#define DEAD        0    		/* if photon is to be terminated */
#define THRESHOLD   0.01		/* used in roulette */
#define CHANCE      0.1  		/* used in roulette */
#define MIN_VALUE	1E-4
#define MIN_WEIGHT	1E-12
#define SQR(x)		(x*x) 
#define SIGN(x)     ((x)>=0.0 ? 1.0:-1.0)
#define COS90D      1.0E-6          /* If cos(theta) <= COS90D, theta >= PI/2 - 1e-6 rad. */
#define ONE_MINUS_COSZERO 1.0E-12   /* If 1-cos(theta) <= ONE_MINUS_COSZERO, fabs(theta) <= 1e-6 rad. */
#define MC_ZERO 0.0
#define MC_ONE 1.0
#define MC_TWO 2.0
#define RandomNum   curand_uniform(&RNGstate) /* Calls for a random number. */

#define GPUID 0
#define RNG_OFFSET 5234567890
#define PHOTONS 1000000
#define MAXSCATT 1000

typedef bool			   GPU_BOOL;
typedef int				   GPU_INT32;
typedef long int		   GPU_LONGINT32;
typedef unsigned int	   GPU_UINT32;
typedef unsigned long long GPU_UINT64;
typedef float GPU_FLOAT;

/* Propagation parameters */
typedef struct tagPhoton
{
public:
	__host__ __device__ tagPhoton()
	{ 
		x = MC_ZERO; y = MC_ZERO; z = MC_ZERO;
		ux = MC_ZERO; uy = MC_ZERO; uz = MC_ZERO;
		uxx = MC_ZERO; uyy = MC_ZERO; uzz = MC_ZERO;
		s = MC_ZERO; sleft = MC_ZERO; costheta = MC_ZERO;
		sintheta = MC_ZERO; cospsi = MC_ZERO; sinpsi = MC_ZERO;
		psi = MC_ZERO; num_scatt = 0; W = MC_ONE; absorb = MC_ZERO;
		photon_status = ALIVE; sv = false;
	};
	double	x, y, z;        /* photon position */
	double	ux, uy, uz;     /* photon trajectory as cosines */
	double  uxx, uyy, uzz;	/* temporary values used during SPIN */
	double	s;              /* step sizes. s = -log(RND)/mus [cm] */
	double  sleft;          /* dimensionless */
	double	costheta;       /* cos(theta) */
	double  sintheta;       /* sin(theta) */
	double	cospsi;         /* cos(psi) */
	double  sinpsi;         /* sin(psi) */
	double	psi;            /* azimuthal angle */
	long	num_scatt;       /* current photon */
	double	W;              /* photon weight */
	double	absorb;         /* weighted deposited in a step due to absorption */
	short   photon_status;  /* flag = ALIVE=1 or DEAD=0 */
	bool sv;             /* Are they in the same voxel? */
} Photon;

/* Run parameters */
typedef struct tagRunParams
{
public:
	__host__ __device__ tagRunParams(){ };
	char   	myname[STRLEN];		// Holds the user's choice of myname, used in input and output files. 
	/* launch parameters */
	int		mcflag, launchflag, boundaryflag;
	float	xfocus, yfocus, zfocus;
	float	ux0, uy0, uz0;
	float	radius;
	float	waist;

	/* mcxyz bin variables */
	float	dx, dy, dz;     /* bin size [cm] */
	int		Nx, Ny, Nz, Nt; /* # of bins */
	float	xs, ys, zs;		/* launch position */

	/* time */
	float	time_min;               // Requested time duration of computation.
} RunParams;

__constant__ RunParams runParamsG;

typedef struct tagTissueParams
{
public:
	__host__ __device__ tagTissueParams(){ };
	/* tissue parameters */
	float 	muav[Ntiss];            // muav[0:Ntiss-1], absorption coefficient of ith tissue type
	float 	musv[Ntiss];            // scattering coeff. 
	float 	gv[Ntiss];              // anisotropy of scattering
} TissueParams;

__constant__ TissueParams tissParamsG;

/* This kernel initializes the Random Number Generator */
__global__ void SetupCurand(curandState *state, GPU_UINT64 seed)
{
	GPU_UINT32 tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, tid, 0, &state[tid]);
}

void ReadRunParams(const char * argv[], RunParams& runParams, TissueParams& tissParams)
{
	/* Input/Output */
	char	filename[STRLEN];   // temporary filename for writing output.
	FILE*	fid = NULL;               // file ID pointer 
	char    buf[32];                // buffer for reading header.dat

	strcpy(runParams.myname, argv[1]);    // acquire name from argument of function call by user.
	printf("name = %s\n", runParams.myname);

	/**** INPUT FILES *****/
	/* IMPORT myname_H.mci */
	strcpy(filename, runParams.myname);
	strcat(filename, "_H.mci");
	fid = fopen(filename, "r");
	fgets(buf, 32, fid);
	// run parameters
	sscanf(buf, "%f", &runParams.time_min); // desired time duration of run [min]
	fgets(buf, 32, fid);
	sscanf(buf, "%d", &runParams.Nx);  // # of bins  
	fgets(buf, 32, fid);
	sscanf(buf, "%d", &runParams.Ny);  // # of bins
	fgets(buf, 32, fid);
	sscanf(buf, "%d", &runParams.Nz);  // # of bins   

	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.dx);	 // size of bins [cm]
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.dy);	 // size of bins [cm] 
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.dz);	 // size of bins [cm] 

	// launch parameters
	fgets(buf, 32, fid);
	sscanf(buf, "%d", &runParams.mcflag);  // mcflag, 0 = uniform, 1 = Gaussian, 2 = iso-pt
	fgets(buf, 32, fid);
	sscanf(buf, "%d", &runParams.launchflag);  // launchflag, 0 = ignore, 1 = manually set
	fgets(buf, 32, fid);
	sscanf(buf, "%d", &runParams.boundaryflag);  // 0 = no boundaries, 1 = escape at all boundaries, 2 = escape at surface only

	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.xs);  // initial launch point
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.ys);  // initial launch point 
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.zs);  // initial launch point

	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.xfocus);  // xfocus
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.yfocus);  // yfocus
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.zfocus);  // zfocus

	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.ux0);  // ux trajectory
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.uy0);  // uy trajectory
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.uz0);  // uz trajectory

	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.radius);  // radius
	fgets(buf, 32, fid);
	sscanf(buf, "%f", &runParams.waist);  // waist

	// tissue optical properties
	fgets(buf, 32, fid);
	sscanf(buf, "%d", &runParams.Nt);				// # of tissue types in tissue list
	for (int i = 1; i <= runParams.Nt; i++) {
		fgets(buf, 32, fid);
		sscanf(buf, "%f", &tissParams.muav[i]);	// absorption coeff [cm^-1]
		fgets(buf, 32, fid);
		sscanf(buf, "%f", &tissParams.musv[i]);	// scattering coeff [cm^-1]
		fgets(buf, 32, fid);
		sscanf(buf, "%f", &tissParams.gv[i]);		// anisotropy of scatter [dimensionless]
	}
	fclose(fid);

}

char* ImportBinaryTissueFile(RunParams& runParams, TissueParams& tissParams)
{
	char *v = NULL;
	int NN = runParams.Nx*runParams.Ny*runParams.Nz;
	v = (char*)malloc(NN*sizeof(char));  /* tissue structure */
	char	filename[STRLEN];   // temporary filename for writing output.
	FILE*	fid = NULL;               // file ID pointer 

	// read binary file
	strcpy(filename, runParams.myname);
	strcat(filename, "_T.bin");
	fid = fopen(filename, "rb");
	fread(v, sizeof(char), NN, fid);
	fclose(fid);
	
	return(v);
}

void PrintRunParameters(const RunParams& runParams, const TissueParams& tissParams, char*v)
{
	printf("time_min = %0.2f min\n", runParams.time_min);
	printf("Nx = %d, dx = %0.4f [cm]\n", runParams.Nx, runParams.dx);
	printf("Ny = %d, dy = %0.4f [cm]\n", runParams.Ny, runParams.dy);
	printf("Nz = %d, dz = %0.4f [cm]\n", runParams.Nz, runParams.dz);

	printf("xs = %0.4f [cm]\n", runParams.xs);
	printf("ys = %0.4f [cm]\n", runParams.ys);
	printf("zs = %0.4f [cm]\n", runParams.zs);
	printf("mcflag = %d [cm]\n", runParams.mcflag);
	if (runParams.mcflag == 0) printf("launching uniform flat-field beam\n");
	if (runParams.mcflag == 1) printf("launching Gaissian beam\n");
	if (runParams.mcflag == 2) printf("launching isotropic point source\n");
	printf("xfocus = %0.4f [cm]\n", runParams.xfocus);
	printf("yfocus = %0.4f [cm]\n", runParams.yfocus);
	printf("zfocus = %0.2e [cm]\n", runParams.zfocus);
	if (runParams.launchflag == 1) {
		printf("Launchflag ON, so launch the following:\n");
		printf("ux0 = %0.4f [cm]\n", runParams.ux0);
		printf("uy0 = %0.4f [cm]\n", runParams.uy0);
		printf("uz0 = %0.4f [cm]\n", runParams.uz0);
	}
	else {
		printf("Launchflag OFF, so program calculates launch angles.\n");
		printf("radius = %0.4f [cm]\n", runParams.radius);
		printf("waist  = %0.4f [cm]\n", runParams.waist);
	}
	if (runParams.boundaryflag == 0)
		printf("boundaryflag = 0, so no boundaries.\n");
	else if (runParams.boundaryflag == 1)
		printf("boundaryflag = 1, so escape at all boundaries.\n");
	else if (runParams.boundaryflag == 2)
		printf("boundaryflag = 2, so escape at surface only.\n");
	else {
		printf("improper boundaryflag. quit.\n");
		throw 0;
	}
	printf("# of tissues available, Nt = %d\n", runParams.Nt);
	for (int i = 1; i <= runParams.Nt; i++) {
		printf("muav[%ld] = %0.4f [cm^-1]\n", i, tissParams.muav[i]);
		printf("musv[%ld] = %0.4f [cm^-1]\n", i, tissParams.musv[i]);
		printf("  gv[%ld] = %0.4f [--]\n\n", i, tissParams.gv[i]);
	}

	// Show tissue on screen, along central z-axis, by listing tissue type #'s.
	int iy = runParams.Ny / 2;
	int ix = runParams.Nx / 2;
	printf("central axial profile of tissue types:\n");
	for (int iz = 0; iz<runParams.Nz; iz++) {
		int i = (long)(iz*runParams.Ny*runParams.Nx + ix*runParams.Ny + iy);
		printf("%d", v[i]);
	}
	printf("\n\n");
}

void SaveOpticalProperties(const RunParams& runParams, const TissueParams& tissParams)
{
	/* Input/Output */
	char	filename[STRLEN];   // temporary filename for writing output.
	FILE*	fid = NULL;               // file ID pointer 
	// SAVE optical properties, for later use by MATLAB.
	strcpy(filename, runParams.myname);
	strcat(filename, "_props.m");
	fid = fopen(filename, "w");
	for (int i = 1; i <= runParams.Nt; i++) {
		fprintf(fid, "muav(%ld) = %0.4f;\n", i, tissParams.muav[i]);
		fprintf(fid, "musv(%ld) = %0.4f;\n", i, tissParams.musv[i]);
		fprintf(fid, "gv(%ld) = %0.4f;\n\n", i, tissParams.gv[i]);
	}
	fclose(fid);
}

/* Compare two doubles in GPU memory */
__device__ __host__ inline GPU_BOOL DblEq(double dbl1, double dbl2, int error = 1)
{
	double errorDbl2 = dbl2 * DBL_EPSILON * (1 + error);
	return (dbl1 >= dbl2 - errorDbl2) && (dbl1 <= dbl2 + errorDbl2);
}

/* Compare two floats in GPU memory */
__device__ __host__ inline GPU_BOOL FltEq(float dbl1, float dbl2, int error = 1)
{
	float errorDbl2 = dbl2 * FLT_EPSILON * (1 + error);
	return (dbl1 >= dbl2 - errorDbl2) && (dbl1 <= dbl2 + errorDbl2);
}


/***********************************************************
* max2
****/
__device__ __host__ inline double max2(double a, double b) {
	double m;
	if (a > b)
		m = a;
	else
		m = b;
	return m;
}

/***********************************************************
* min2
****/
__device__ __host__ inline double min2(double a, double b) {
	double m;
	if (a >= b)
		m = b;
	else
		m = a;
	return m;
}
/***********************************************************
* min3
****/
__device__ __host__ inline double min3(double a, double b, double c) {
	double m;
	if (a <= min2(b, c))
		m = a;
	else if (b <= min2(a, c))
		m = b;
	else
		m = c;
	return m;
}

/***********************************************************
*  Determine if the two position are located in the same voxel
*	Returns 1 if same voxel, 0 if not same voxel.
****/
__device__ __host__ inline bool SameVoxel(double x1, double y1, double z1, double x2, double y2, double z2, double dx, double dy, double dz)
{
	double xmin = min2((floor)(x1 / dx), (floor)(x2 / dx))*dx;
	double ymin = min2((floor)(y1 / dy), (floor)(y2 / dy))*dy;
	double zmin = min2((floor)(z1 / dz), (floor)(z2 / dz))*dz;
	double xmax = xmin + dx;
	double ymax = ymin + dy;
	double zmax = zmin + dz;
	bool sv = 0;

	sv = (x1 <= xmax && x2 <= xmax && y1 <= ymax && y2 <= ymax && z1<zmax && z2 <= zmax);
	return (sv);
}


/********************
* my version of FindVoxelFace for no scattering.
* s = ls + FindVoxelFace2(x,y,z, tempx, tempy, tempz, dx, dy, dz, ux, uy, uz);
****/
__device__ __host__ inline double FindVoxelFace2(double x1, double y1, double z1, double x2, double y2, double z2, double dx, double dy, double dz, double ux, double uy, double uz)
{
	int ix1 = (int)floor(x1 / dx);
	int iy1 = (int)floor(y1 / dy);
	int iz1 = (int)floor(z1 / dz);

	int ix2, iy2, iz2;
	if (ux >= 0.0)
		ix2 = ix1 + 1;
	else
		ix2 = ix1;

	if (uy >= 0.0)
		iy2 = iy1 + 1;
	else
		iy2 = iy1;

	if (uz >= 0.0)
		iz2 = iz1 + 1;
	else
		iz2 = iz1;

	double xs = fabs((ix2*dx - x1) / ux);
	double ys = fabs((iy2*dy - y1) / uy);
	double zs = fabs((iz2*dz - z1) / uz);

	double s = min3(xs, ys, zs);

	return (s);
}


/***********************************************************
*	FRESNEL REFLECTANCE
* Computes reflectance as photon passes from medium 1 to
* medium 2 with refractive indices n1,n2. Incident
* angle a1 is specified by cosine value ca1 = cos(a1).
* Program returns value of transmitted angle a1 as
* value in *ca2_Ptr = cos(a2).
****/
__device__ __host__ inline double RFresnel(double n1,		/* incident refractive index.*/
	double n2,		/* transmit refractive index.*/
	double ca1,		/* cosine of the incident */
	/* angle a1, 0<a1<90 degrees. */
	double *ca2_Ptr) 	/* pointer to the cosine */
	/* of the transmission */
	/* angle a2, a2>0. */
{
	double r;

	if (n1 == n2) { /** matched boundary. **/
		*ca2_Ptr = ca1;
		r = 0.0;
	}
	else if (ca1>(1.0 - 1.0e-12)) { /** normal incidence. **/
		*ca2_Ptr = ca1;
		r = (n2 - n1) / (n2 + n1);
		r *= r;
	}
	else if (ca1< 1.0e-6) {	/** very slanted. **/
		*ca2_Ptr = 0.0;
		r = 1.0;
	}
	else {			  		/** general. **/
		double sa1, sa2; /* sine of incident and transmission angles. */
		double ca2;      /* cosine of transmission angle. */
		sa1 = sqrt(1 - ca1*ca1);
		sa2 = n1*sa1 / n2;
		if (sa2 >= 1.0) {
			/* double check for total internal reflection. */
			*ca2_Ptr = 0.0;
			r = 1.0;
		}
		else {
			double cap, cam;	/* cosines of sum ap or diff am of the two */
			/* angles: ap = a1 + a2, am = a1 - a2. */
			double sap, sam;	/* sines. */
			*ca2_Ptr = ca2 = sqrt(1 - sa2*sa2);
			cap = ca1*ca2 - sa1*sa2; /* c+ = cc - ss. */
			cam = ca1*ca2 + sa1*sa2; /* c- = cc + ss. */
			sap = sa1*ca2 + ca1*sa2; /* s+ = sc + cs. */
			sam = sa1*ca2 - ca1*sa2; /* s- = sc - cs. */
			r = 0.5*sam*sam*(cam*cam + cap*cap) / (sap*sap*cam*cam);
			/* rearranged for speed. */
		}
	}
	return(r);
} /******** END SUBROUTINE **********/



/***********************************************************
* the boundary is the face of some voxel
* find the the photon's hitting position on the nearest face of the voxel and update the step size.
****/
__device__ __host__ inline double FindVoxelFace(double x1, double y1, double z1, double x2, double y2, double z2, double dx, double dy, double dz, double ux, double uy, double uz)
{
	double x_1 = x1 / dx;
	double y_1 = y1 / dy;
	double z_1 = z1 / dz;
	double x_2 = x2 / dx;
	double y_2 = y2 / dy;
	double z_2 = z2 / dz;
	double fx_1 = floor(x_1);
	double fy_1 = floor(y_1);
	double fz_1 = floor(z_1);
	double fx_2 = floor(x_2);
	double fy_2 = floor(y_2);
	double fz_2 = floor(z_2);
	double x = 0.0, y = 0.0, z = 0.0, x0 = 0.0, y0 = 0.0, z0 = 0.0, s = 0.0;

	if ((fx_1 != fx_2) && (fy_1 != fy_2) && (fz_1 != fz_2)) { //#10
		fx_2 = fx_1 + SIGN(fx_2 - fx_1);//added
		fy_2 = fy_1 + SIGN(fy_2 - fy_1);//added
		fz_2 = fz_1 + SIGN(fz_2 - fz_1);//added

		x = (max2(fx_1, fx_2) - x_1) / ux;
		y = (max2(fy_1, fy_2) - y_1) / uy;
		z = (max2(fz_1, fz_2) - z_1) / uz;
		if (x == min3(x, y, z)) {
			x0 = max2(fx_1, fx_2);
			y0 = (x0 - x_1) / ux*uy + y_1;
			z0 = (x0 - x_1) / ux*uz + z_1;
		}
		else if (y == min3(x, y, z)) {
			y0 = max2(fy_1, fy_2);
			x0 = (y0 - y_1) / uy*ux + x_1;
			z0 = (y0 - y_1) / uy*uz + z_1;
		}
		else {
			z0 = max2(fz_1, fz_2);
			y0 = (z0 - z_1) / uz*uy + y_1;
			x0 = (z0 - z_1) / uz*ux + x_1;
		}
	}
	else if ((fx_1 != fx_2) && (fy_1 != fy_2)) { //#2
		fx_2 = fx_1 + SIGN(fx_2 - fx_1);//added
		fy_2 = fy_1 + SIGN(fy_2 - fy_1);//added
		x = (max2(fx_1, fx_2) - x_1) / ux;
		y = (max2(fy_1, fy_2) - y_1) / uy;
		if (x == min2(x, y)) {
			x0 = max2(fx_1, fx_2);
			y0 = (x0 - x_1) / ux*uy + y_1;
			z0 = (x0 - x_1) / ux*uz + z_1;
		}
		else {
			y0 = max2(fy_1, fy_2);
			x0 = (y0 - y_1) / uy*ux + x_1;
			z0 = (y0 - y_1) / uy*uz + z_1;
		}
	}
	else if ((fy_1 != fy_2) && (fz_1 != fz_2)) { //#3
		fy_2 = fy_1 + SIGN(fy_2 - fy_1);//added
		fz_2 = fz_1 + SIGN(fz_2 - fz_1);//added
		y = (max2(fy_1, fy_2) - y_1) / uy;
		z = (max2(fz_1, fz_2) - z_1) / uz;
		if (y == min2(y, z)) {
			y0 = max2(fy_1, fy_2);
			x0 = (y0 - y_1) / uy*ux + x_1;
			z0 = (y0 - y_1) / uy*uz + z_1;
		}
		else {
			z0 = max2(fz_1, fz_2);
			x0 = (z0 - z_1) / uz*ux + x_1;
			y0 = (z0 - z_1) / uz*uy + y_1;
		}
	}
	else if ((fx_1 != fx_2) && (fz_1 != fz_2)) { //#4
		fx_2 = fx_1 + SIGN(fx_2 - fx_1);//added
		fz_2 = fz_1 + SIGN(fz_2 - fz_1);//added
		x = (max2(fx_1, fx_2) - x_1) / ux;
		z = (max2(fz_1, fz_2) - z_1) / uz;
		if (x == min2(x, z)) {
			x0 = max2(fx_1, fx_2);
			y0 = (x0 - x_1) / ux*uy + y_1;
			z0 = (x0 - x_1) / ux*uz + z_1;
		}
		else {
			z0 = max2(fz_1, fz_2);
			x0 = (z0 - z_1) / uz*ux + x_1;
			y0 = (z0 - z_1) / uz*uy + y_1;
		}
	}
	else if (fx_1 != fx_2) { //#5
		fx_2 = fx_1 + SIGN(fx_2 - fx_1);//added
		x0 = max2(fx_1, fx_2);
		y0 = (x0 - x_1) / ux*uy + y_1;
		z0 = (x0 - x_1) / ux*uz + z_1;
	}
	else if (fy_1 != fy_2) { //#6
		fy_2 = fy_1 + SIGN(fy_2 - fy_1);//added
		y0 = max2(fy_1, fy_2);
		x0 = (y0 - y_1) / uy*ux + x_1;
		z0 = (y0 - y_1) / uy*uz + z_1;
	}
	else { //#7 
		z0 = max2(fz_1, fz_2);
		fz_2 = fz_1 + SIGN(fz_2 - fz_1);//added
		x0 = (z0 - z_1) / uz*ux + x_1;
		y0 = (z0 - z_1) / uz*uy + y_1;
	}
	//s = ( (x0-fx_1)*dx + (y0-fy_1)*dy + (z0-fz_1)*dz )/3;
	//s = sqrt( SQR((x0-x_1)*dx) + SQR((y0-y_1)*dy) + SQR((z0-z_1)*dz) );
	//s = sqrt(SQR(x0-x_1)*SQR(dx) + SQR(y0-y_1)*SQR(dy) + SQR(z0-z_1)*SQR(dz));
	s = sqrt(SQR((x0 - x_1)*dx) + SQR((y0 - y_1)*dy) + SQR((z0 - z_1)*dz));
	return (s);
}

/* This kernel simulates photon propagation */
__global__ void PerformMCXYZ(curandState *devRNGstate, const char* v, float* devResultsArray, GPU_UINT64 *photons_dev)
{
	GPU_UINT32 tid = threadIdx.x + blockIdx.x * blockDim.x; 
	curandState RNGstate = devRNGstate[tid];

	Photon new_photon;

	/**** LAUNCH
	Initialize photon position and trajectory.
	*****/
	atomicAdd(photons_dev, 1);
	
	new_photon.num_scatt = 0;
	new_photon.W = 1.0;                    /* set photon weight to one */
	new_photon.photon_status = ALIVE;      /* Launch an ALIVE photon */

	/**** SET SOURCE
	* Launch collimated beam at x,y center.
	****/

	/****************************/
	/* Initial position. */

	/* trajectory */
	if (runParamsG.launchflag == 1) { // manually set launch
		new_photon.x = runParamsG.xs;
		new_photon.y = runParamsG.ys;
		new_photon.z = runParamsG.zs;
		new_photon.ux = runParamsG.ux0;
		new_photon.uy = runParamsG.uy0;
		new_photon.uz = runParamsG.uz0;
	}
	else { // use mcflag
		if (runParamsG.mcflag == 0) { // uniform beam
			// set launch point and width of beam
			GPU_FLOAT rnd = RandomNum;
			GPU_FLOAT r = runParamsG.radius*sqrt(rnd); // radius of beam at launch point
			rnd = RandomNum;
			GPU_FLOAT phi = MC_TWO * PI * rnd;
			new_photon.x = runParamsG.xs + r*cos(phi);
			new_photon.y = runParamsG.ys + r*sin(phi);
			new_photon.z = runParamsG.zs;
			// set trajectory toward focus
			rnd = RandomNum;
			r = runParamsG.waist*sqrt(rnd); // radius of beam at focus
			phi = MC_TWO * PI * RandomNum;
			double xfocus = r*cos(phi);
			double yfocus = r*sin(phi);
			GPU_FLOAT temp = sqrt((new_photon.x - xfocus)*(new_photon.x - xfocus) + (new_photon.y - yfocus)*(new_photon.y - yfocus) + runParamsG.zfocus*runParamsG.zfocus);
			new_photon.ux = -(new_photon.x - xfocus) / temp;
			new_photon.uy = -(new_photon.y - runParamsG.yfocus) / temp;
			new_photon.uz = sqrt(1.0 - new_photon.ux*new_photon.ux + new_photon.uy*new_photon.uy);
		}
		else { // isotropic pt source
			new_photon.costheta = 1.0 - 2.0*RandomNum;
			new_photon.sintheta = sqrt(1.0 - new_photon.costheta*new_photon.costheta);
			new_photon.psi = 2.0*PI*RandomNum;
			new_photon.cospsi = cos(new_photon.psi);
			if (new_photon.psi < PI)
				new_photon.sinpsi = sqrt(1.0 - new_photon.cospsi*new_photon.cospsi);
			else
				new_photon.sinpsi = -sqrt(1.0 - new_photon.cospsi*new_photon.cospsi);
			new_photon.x = runParamsG.xs;
			new_photon.y = runParamsG.ys;
			new_photon.z = runParamsG.zs;
			new_photon.ux = new_photon.sintheta*new_photon.cospsi;
			new_photon.uy = new_photon.sintheta*new_photon.sinpsi;
			new_photon.uz = new_photon.costheta;
			}
		} // end  use mcflag
		/****************************/

		/* Get tissue voxel properties of launchpoint.
		* If photon beyond outer edge of defined voxels,
		* the tissue equals properties of outermost voxels.
		* Therefore, set outermost voxels to infinite background value.
		*/
		 /* Added. Used to track photons */
		int ix = (int)(runParamsG.Nx / 2 + new_photon.x / runParamsG.dx);
		int iy = (int)(runParamsG.Ny / 2 + new_photon.y / runParamsG.dy);
		int iz = (int)(new_photon.z / runParamsG.dz);
		if (ix >= runParamsG.Nx) ix = runParamsG.Nx - 1;
		if (iy >= runParamsG.Ny) iy = runParamsG.Ny - 1;
		if (iz >= runParamsG.Nz) iz = runParamsG.Nz - 1;
		if (ix<0)   ix = 0;
		if (iy<0)   iy = 0;
		if (iz<0)   iz = 0;
		/* Get the tissue type of located voxel */
		long i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
		int type = v[i];
		double mua = tissParamsG.muav[type];
		double mus = tissParamsG.musv[type];
		double g = tissParamsG.gv[type];

		int bflag = 1; // initialize as 1 = inside volume, but later check as photon propagates.

		/* HOP_DROP_SPIN_CHECK
		Propagate one photon until it dies as determined by ROULETTE.
		*******/

		do {

			/**** HOP
			Take step to new position
			s = dimensionless stepsize
			x, uy, uz are cosines of current photon trajectory
			*****/
			GPU_FLOAT rnd = RandomNum;  /* yields 0 < rnd <= 1 */
			new_photon.sleft = -log(rnd);				/* dimensionless step */

			do {  // while sleft>0   
				new_photon.s = new_photon.sleft / mus;				/* Step size [cm].*/
				double	tempx, tempy, tempz; /* temporary variables, used during photon step. */
				tempx = new_photon.x + new_photon.s*new_photon.ux;				/* Update positions. [cm] */
				tempy = new_photon.y + new_photon.s*new_photon.uy;
				tempz = new_photon.z + new_photon.s*new_photon.uz;
				
				new_photon.sv = SameVoxel(new_photon.x, new_photon.y, new_photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz);
				
				if (new_photon.sv) /* photon in same voxel */
				{
					new_photon.x = tempx;					/* Update positions. */
					new_photon.y = tempy;
					new_photon.z = tempz;

					/**** DROP
					Drop photon weight (W) into local bin.
					*****/
					new_photon.absorb = new_photon.W*(1.0 - exp(-mua*new_photon.s));	/* photon weight absorbed at this step */
					new_photon.W -= new_photon.absorb;					/* decrement WEIGHT by amount absorbed */
					// If photon within volume of heterogeneity, deposit energy in F[]. 
					// Normalize F[] later, when save output. 
					if (bflag && !isnan(new_photon.absorb) && !isinf(new_photon.absorb))
						atomicAdd(&devResultsArray[i], (float)new_photon.absorb); // only save data if blag==1, i.e., photon inside simulation cube

					/* Update sleft */
					new_photon.sleft = 0.0;		/* dimensionless step remaining */
				}
				else /* photon has crossed voxel boundary */
				{
					/* step to voxel face + "littlest step" so just inside new voxel. */
					new_photon.s = ls + FindVoxelFace2(new_photon.x, new_photon.y, new_photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz, new_photon.ux, new_photon.uy, new_photon.uz);

					/**** DROP
					Drop photon weight (W) into local bin.
					*****/
					new_photon.absorb = new_photon.W*(1.0 - exp(-mua*new_photon.s));   /* photon weight absorbed at this step */
					new_photon.W -= new_photon.absorb;                  /* decrement WEIGHT by amount absorbed */
					// If photon within volume of heterogeneity, deposit energy in F[]. 
					// Normalize F[] later, when save output. 
					if (bflag && !isnan(new_photon.absorb) && !isinf(new_photon.absorb))
						atomicAdd(&devResultsArray[i], (float)new_photon.absorb); // only save data if blag==1, i.e., photon inside simulation cube

					/* Update sleft */
					new_photon.sleft -= new_photon.s*mus;  /* dimensionless step remaining */
					if (new_photon.sleft <= ls) new_photon.sleft = 0.0;

					/* Update positions. */
					new_photon.x += new_photon.s*new_photon.ux;
					new_photon.y += new_photon.s*new_photon.uy;
					new_photon.z += new_photon.s*new_photon.uz;

					// pointers to voxel containing optical properties
					ix = (int)(runParamsG.Nx / 2 + new_photon.x / runParamsG.dx);
					iy = (int)(runParamsG.Ny / 2 + new_photon.y / runParamsG.dy);
					iz = (int)(new_photon.z / runParamsG.dz);

					bflag = 1;  // Boundary flag. Initialize as 1 = inside volume, then check.
					if (runParamsG.boundaryflag == 0) { // Infinite medium.
						// Check if photon has wandered outside volume.
						// If so, set tissue type to boundary value, but let photon wander.
						// Set blag to zero, so DROP does not deposit energy.
						if (iz >= runParamsG.Nz) { iz = runParamsG.Nz - 1; bflag = 0; }
						if (ix >= runParamsG.Nx) { ix = runParamsG.Nx - 1; bflag = 0; }
						if (iy >= runParamsG.Ny) { iy = runParamsG.Ny - 1; bflag = 0; }
						if (iz<0) { iz = 0;    bflag = 0; }
						if (ix<0) { ix = 0;    bflag = 0; }
						if (iy<0) { iy = 0;    bflag = 0; }
					}
					else if (runParamsG.boundaryflag == 1) { // Escape at boundaries
						if (iz >= runParamsG.Nz) { iz = runParamsG.Nz - 1; new_photon.photon_status = DEAD; new_photon.sleft = 0; }
						if (ix >= runParamsG.Nx) { ix = runParamsG.Nx - 1; new_photon.photon_status = DEAD; new_photon.sleft = 0; }
						if (iy >= runParamsG.Ny) { iy = runParamsG.Ny - 1; new_photon.photon_status = DEAD; new_photon.sleft = 0; }
						if (iz<0) { iz = 0;    new_photon.photon_status = DEAD; new_photon.sleft = 0; }
						if (ix<0) { ix = 0;    new_photon.photon_status = DEAD; new_photon.sleft = 0; }
						if (iy<0) { iy = 0;    new_photon.photon_status = DEAD; new_photon.sleft = 0; }
					}
					else if (runParamsG.boundaryflag == 2) { // Escape at top surface, no x,y bottom z boundaries
						if (iz >= runParamsG.Nz) { iz = runParamsG.Nz - 1; bflag = 0; }
						if (ix >= runParamsG.Nx) { ix = runParamsG.Nx - 1; bflag = 0; }
						if (iy >= runParamsG.Ny) { iy = runParamsG.Ny - 1; bflag = 0; }
						if (iz<0) { iz = 0;    new_photon.photon_status = DEAD; new_photon.sleft = 0; }
						if (ix<0) { ix = 0;    bflag = 0; }
						if (iy<0) { iy = 0;    bflag = 0; }
					}

					// update pointer to tissue type
					i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
					type = v[i];
					mua = tissParamsG.muav[type];
					mus = tissParamsG.musv[type];
					g = tissParamsG.gv[type];

				} //(sv) /* same voxel */

			} while (new_photon.sleft>0.0); //do...while

		
			/**** SPIN
			Scatter photon into new trajectory defined by theta and psi.
			Theta is specified by cos(theta), which is determined
			based on the Henyey-Greenstein scattering function.
			Convert theta and psi into cosines ux, uy, uz.
			*****/
			/* Sample for costheta */
			rnd = RandomNum;
			if (DblEq(g,0.0))
				new_photon.costheta = 2.0*rnd - 1.0;
			else {
				double temp = (1.0 - g*g) / (1.0 - g + 2.0 * g*rnd);
				new_photon.costheta = (1.0 + g*g - temp*temp) / (2.0*g);
			}
			new_photon.sintheta = sqrt(1.0 - new_photon.costheta*new_photon.costheta); /* sqrt() is faster than sin(). */

			/* Sample psi. */
			new_photon.psi = 2.0*PI*RandomNum;
			new_photon.cospsi = cos(new_photon.psi);
			if (new_photon.psi < PI)
				new_photon.sinpsi = sqrt(1.0 - new_photon.cospsi*new_photon.cospsi);     /* sqrt() is faster than sin(). */
			else
				new_photon.sinpsi = -sqrt(1.0 - new_photon.cospsi*new_photon.cospsi);

			/* New trajectory. */
			if (1.0 - fabs(new_photon.uz) <= ONE_MINUS_COSZERO) {      /* close to perpendicular. */
				new_photon.uxx = new_photon.sintheta * new_photon.cospsi;
				new_photon.uyy = new_photon.sintheta * new_photon.sinpsi;
				new_photon.uzz = new_photon.costheta * SIGN(new_photon.uz);   /* SIGN() is faster than division. */
			}
			else {					/* usually use this option */
				double temp = sqrt(1.0 - new_photon.uz * new_photon.uz);
				new_photon.uxx = new_photon.sintheta * (new_photon.ux * new_photon.uz * new_photon.cospsi - new_photon.uy * new_photon.sinpsi) / temp + new_photon.ux * new_photon.costheta;
				new_photon.uyy = new_photon.sintheta * (new_photon.uy * new_photon.uz * new_photon.cospsi + new_photon.ux * new_photon.sinpsi) / temp + new_photon.uy * new_photon.costheta;
				new_photon.uzz = -new_photon.sintheta * new_photon.cospsi * temp + new_photon.uz * new_photon.costheta;
			}

			/* Update trajectory */
			new_photon.ux = new_photon.uxx;
			new_photon.uy = new_photon.uyy;
			new_photon.uz = new_photon.uzz;
			new_photon.num_scatt++;
			
			/**** CHECK ROULETTE
			If photon weight below THRESHOLD, then terminate photon using Roulette technique.
			Photon has CHANCE probability of having its weight increased by factor of 1/CHANCE,
			and 1-CHANCE probability of terminating.
			*****/
			if (new_photon.W < THRESHOLD) {
				if ((MC_TWO * curand_uniform(&RNGstate) - MC_ONE) <= CHANCE)
					new_photon.W /= CHANCE;
				else new_photon.photon_status = DEAD;
			}
			
			// Russian roulette with a semi automatic
			if (new_photon.W < MIN_VALUE || new_photon.num_scatt > MAXSCATT)
				new_photon.photon_status = DEAD;

		} while (new_photon.photon_status == ALIVE);  /* end STEP_CHECK_HOP_SPIN */
		/* if ALIVE, continue propagating */
		/* If photon DEAD, then launch new photon. */


	devRNGstate[tid] = RNGstate; 
}

int main(int argc, const char * argv[]) {

	if (argc == 0) {
		printf("assuming you've compiled mcxyz.c as gomcxyz ...\n");
		printf("USAGE: gomcxyz name\n");
		printf("which will load the files name_H.mci and name_T.bin\n");
		printf("and run the Monte Carlo program.\n");
		printf("Yields  name_F.bin, which holds the fluence rate distribution.\n");
		return 0;
	}
	// Simulation Parameters
	RunParams runParams;
	TissueParams tissParams;
	ReadRunParams(argv, runParams, tissParams);
	char* v = ImportBinaryTissueFile(runParams, tissParams);
	PrintRunParameters(runParams, tissParams, v);
	SaveOpticalProperties(runParams, tissParams);

	// Prepare GPU and event recording
	cudaSetDevice(GPUID);
	cudaDeviceReset();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Setup memory chain
	cudaError err_t;
	int NN = runParams.Nx*runParams.Ny*runParams.Nz;
	char* devV = NULL;
	cudaMalloc((void**)&devV, NN*sizeof(char));
	err_t = cudaGetLastError();
	cudaMemset(devV, 0x0, NN*sizeof(char));
	cudaMemcpy(devV, v, NN*sizeof(char), cudaMemcpyHostToDevice);
	err_t = cudaGetLastError();

	cudaMemcpyToSymbol(tissParamsG, &tissParams, sizeof(TissueParams));
	cudaMemcpyToSymbol(runParamsG, &runParams, sizeof(RunParams));
	err_t = cudaGetLastError();
	
	// Results
	size_t NeedMem = sizeof(float) * NN;
	float* hostF = (float*)malloc(NeedMem);	/* relative fluence rate [W/cm^2/W.delivered] */
	memset(hostF, 0x0, NeedMem);
	
	/* relative fluence rate [W/cm^2/W.delivered] */
	float* devF = NULL;
	cudaMalloc((void**)&devF, NeedMem);
	cudaMemset(devF, 0x0, NeedMem);
	err_t = cudaGetLastError();
	GPU_UINT64* detected_photons_dev = NULL;
	cudaMalloc((void**)&detected_photons_dev, sizeof(GPU_UINT64));
	cudaMemset(detected_photons_dev, 0x0, sizeof(GPU_UINT64));
	err_t = cudaGetLastError();

	// Kernel parameters
	GPU_UINT64 detected_photons = NULL;
	dim3 blocks(BLOCK_DIM_MCXYZ, 1);
	dim3 threads(THREADS_DIM_MCXYZ, 1);

	// Setup CURAND
	curandState *devStates = NULL;
	cudaMalloc((void**)&devStates, PHOTONS_MCXYZ * sizeof(curandState));
	err_t = cudaGetLastError();
	SetupCurand<<<blocks, threads >> >(devStates, RNG_OFFSET);
	cudaThreadSynchronize();
	err_t = cudaGetLastError();

	/* Monte Carlo */
	printf("------------- Begin Monte Carlo -------------\n");
	printf("%s\n", runParams.myname);
	do
	{
		PerformMCXYZ<<<blocks, threads>>>(devStates, devV, devF, detected_photons_dev);
		cudaThreadSynchronize();
		err_t = cudaGetLastError();
		cudaMemcpy(&detected_photons, detected_photons_dev, sizeof(GPU_UINT64), cudaMemcpyDeviceToHost);
		err_t = cudaGetLastError();
		int completion_percent = (int)(detected_photons * 100.0 / PHOTONS);
		printf("GPU Kernel has been executed, %d percent completed \n", completion_percent < 100 ? completion_percent : 100);

	} while (detected_photons < PHOTONS);

	// Results
	cudaMemcpy(hostF, devF, NeedMem, cudaMemcpyDeviceToHost);

	err_t = cudaGetLastError();
	
	// Timing and performace
	float elapsedTime;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("------------------------------------------------------\n");
	printf("Monte Carlo simulation has been perfromed for %lld photons in %.2f seconds or %.2f minutes / %.2f hours\n", detected_photons, elapsedTime / 1000.0, elapsedTime / (1000.0 * 60.0), elapsedTime / (1000.0 * 60.0 * 60.0));
	printf("%d photons per second, %d per minute\n", (int)(detected_photons / (elapsedTime / (1000.0))), (int)(detected_photons / (elapsedTime / (1000.0 * 60.0))));

	/**** SAVE
	Convert data to relative fluence rate [cm^-2] and save.
	*****/

	// Normalize deposition (A) to yield fluence rate (F).
	float temp = runParams.dx*runParams.dy*runParams.dz*detected_photons;
	for (int i = 0; i<NN; i++) {
		float Val = hostF[i] / (temp*tissParams.muav[v[i]]);
		if (!isnan(Val) && !isinf(Val))
			hostF[i] = Val;
		else
			hostF[i] = 0.0F;
	}
	// Save the binary file
	char filename[STRLEN];   // temporary filename for writing output.
	strcpy(filename, runParams.myname);
	strcat(filename, "_F.bin");
	printf("saving %s\n", filename);
	FILE* fid = fopen(filename, "wb");   /* 3D voxel output */
	fwrite(hostF, sizeof(float), NN, fid);
	fclose(fid);

	printf("------------------------------------------------------\n");

	// Clean up
	if (devF)
		cudaFree(devF);
	if (devV)
		cudaFree(devV);
	if (devStates)
		cudaFree(devStates);
	free(v);
	free(hostF);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return 0;
} /* end of main */

