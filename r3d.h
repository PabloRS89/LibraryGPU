/*
 *
 *		r3d.h
 *		
 *		Routines for fast, geometrically robust clipping operations
 *		and analytic volume/moment computations over polyhedra in 3D. 
 *		
 *		Devon Powell
 *		31 August 2015
 *		
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 *    
 */

#ifndef _R3D_H_
#define _R3D_H_

#include <stdint.h>

#define R3D_MAX_VERTS 64
/**
 * \file r3d.h
 * \author Devon Powell
 * \date 31 August 2015
 * \brief Interface for r3d
 */

/**
 * \brief Real type specifying the precision to be used in calculations
 *
 * Default is `double`. `float` precision is enabled by 
 * compiling with `-DSINGLE_PRECISION`.
 */
#ifdef SINGLE_PRECISION
typedef float r3d_real;
#else 
typedef double r3d_real;
#endif

/**
 * \brief Integer types used for indexing
 */
typedef int32_t r3d_int;
typedef int64_t r3d_long;

#endif // _V3D_H_
