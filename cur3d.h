
#ifndef _CUR3D_H_
#define _CUR3D_H_

extern "C" {
	#include "r3d.h"
}

// Constants
#define NUM_SM 14 // num. streaming multiprocessors
#define THREADS_PER_SM 512 // num threads to launch per SM in coarse binning

#endif // _CUR3D_H_