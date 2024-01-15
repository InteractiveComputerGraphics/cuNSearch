#pragma once
#include "Types.h"
#include "PointSet.h"
#include <thrust/device_vector.h>

namespace cuNSearch
{
	class cuNSearchDeviceData
	{
	public:
		cuNSearchDeviceData(Real searchRadius)
		{
			m_SearchRadius = searchRadius;
		}

		void setSearchRadius(Real searchRadius)
		{
			m_SearchRadius = searchRadius;
		}

		/** Compute min max for the given point set
		*/
		void computeMinMax(PointSet &pointSet);


		/** Constructs the uniform grid for the given point set a updates all cell information to allow queries on this point set
		*/
		void computeCellInformation(PointSet &pointSet);


		/** Queries the neighbors in the given point set for all particles in the query point set.
		*/
		void computeNeighborhood(PointSet &queryPointSet, PointSet &pointSet, uint neighborListEntry);

	private:
		Real m_SearchRadius;

		thrust::device_vector<Int3> d_MinMax;
		thrust::device_vector<uint> d_TempSortIndices;

		// only temporary used. After neighborhood computation pointer is handed over to point set
/* 		uint* &d_Neighbors;
		uint* &d_NeighborCounts;
		uint* &d_NeighborWriteOffsets; */
	};
};