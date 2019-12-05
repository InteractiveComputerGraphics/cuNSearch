#pragma once
// This is a public header. Avoid references to cuda or other external references.

#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

#include "Common.h"

namespace cuNSearch
{
class NeighborhoodSearch;
class PointSetImplementation;
class cuNSearchDeviceData;

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) 
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/**
* @class PointSet.
* Represents a set of points in three-dimensional space.
*/
class PointSet
{
	struct NeighborSet
	{
		//Pinned memory
		uint NeighborCountAllocationSize;
		uint ParticleCountAllocationSize;
		uint *Counts;
		uint *Offsets;
		uint *Neighbors;

		NeighborSet()
		{
			NeighborCountAllocationSize = 0u;
			ParticleCountAllocationSize = 0u;
			Counts = nullptr;
			Offsets = nullptr;
			Neighbors = nullptr;
		}
	};

public:
	///**
	//* Copy constructor.
	//*/
	PointSet(PointSet const& other);


	~PointSet();
	//Define descructor in cpp file to allow unique_ptr to incomplete type.
	//https://stackoverflow.com/questions/9954518/stdunique-ptr-with-an-incomplete-type-wont-compile

	/**
	* Returns the number of neighbors of point i in the given point set.
	* @param i Point index.
	* @returns Number of points neighboring point i in point set point_set.
	*/
	inline std::size_t n_neighbors(unsigned int point_set, unsigned int i) const 
	{
		return neighbors[point_set].Counts[i];
	}

	/**
	* Fetches id pair of kth neighbor of point i in the given point set.
	* @param point_set Point set index of other point set where neighbors have been searched.
	* @param i Point index for which the neighbor id should be returned.
	* @param k Represents kth neighbor of point i.
	* @returns Index of neighboring point i in point set point_set.
	*/
	inline unsigned int neighbor(unsigned int point_set, unsigned int i, unsigned int k) const 
	{
		//Return index of the k-th neighbor to point i (of the given point set)
		const auto &neighborSet = neighbors[point_set];
		return neighborSet.Neighbors[neighborSet.Offsets[i] + k];
	}

	inline uint n_neighborsets()
	{
		return neighbors.size();
	}

	inline uint neighbor_count(const uint i)
	{
		return neighbors[i].NeighborCountAllocationSize;
	}

	inline uint particle_count(const uint i)
	{
		return neighbors[i].ParticleCountAllocationSize;
	}

	inline const uint* neighbor_indices(const uint i)
	{
		return neighbors[i].Neighbors;
	}

	inline const uint* neighbor_counts(const uint i)
	{
		return neighbors[i].Counts;
	}

	inline const uint* neighbor_offsets(const uint i)
	{
		return neighbors[i].Offsets;
	}

	PointSetImplementation *getPointSetImplementation()
	{
		return impl.get();
	}

	/**
	* Fetches pointer to neighbors of point i in the given point set.
	* @param point_set Point set index of other point set where neighbors have been searched.
	* @param i Point index for which the neighbor id should be returned.
	* @returns Pointer to ids of neighboring points of i in point set point_set.
	*/
	inline unsigned int * neighbor_list(unsigned int point_set, unsigned int i) const 
	{
		//Return index of the k-th neighbor to point i (of the given point set)
		const auto &neighborSet = neighbors[point_set];
		return &neighborSet.Neighbors[neighborSet.Offsets[i]];
	}
	
	/**
	* @returns the number of points contained in the point set.
	*/
	std::size_t n_points() const { return m_n; }

	/*
	* Returns true, if the point locations may be updated by the user.
	**/
	bool is_dynamic() const { return m_dynamic; }

	/**
	* If true is passed, the point positions may be altered by the user.
	*/
	void set_dynamic(bool v) { m_dynamic = v; }

	Real const* GetPoints() { return m_x; }

	/**
	* Return the user data which can be attached to a point set.
	*/
	void *get_user_data() { return m_user_data; }

	/**
	* Reorders an array according to a previously generated sort table by invocation of the method
	* "z_sort" of class "NeighborhoodSearch". Please note that the method "z_sort" of class
	* "Neighborhood search" has to be called beforehand.
	*/
	template <typename T>
	void sort_field(T* lst) const;

private:
	friend NeighborhoodSearch;
	friend cuNSearchDeviceData;

	// Implementation and cuda data are hidden in the PointSetImplementation class to avoid unnecessary dependencies in public headers.
	std::unique_ptr<PointSetImplementation> impl;

	PointSet(Real const* x, std::size_t n, bool dynamic, void *user_data = nullptr);
	
	void resize(Real const* x, std::size_t n);

	Real const* point(unsigned int i) const { return &m_x[3*i]; }

	Real const* m_x;	//positions of the points
	std::size_t m_n;	//# of points in the set
	bool m_dynamic;		//if false the points do not move and the hash values do not change
	void *m_user_data;

	std::vector<uint> sortIndices;
	std::vector<NeighborSet> neighbors;
};


template <typename T>
void PointSet::sort_field(T* lst) const
{
	std::vector<T> tmp(lst, lst + sortIndices.size());
	std::transform(sortIndices.begin(), sortIndices.end(),
//#ifdef _MSC_VER
//		stdext::unchecked_array_iterator<T*>(lst),
//#else
		lst,
//#endif
		[&](int i) { return tmp[i]; });
}


}

