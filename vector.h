#ifndef VECTOR_H
#define VECTOR_H

#include "math.h"
#include "stdio.h"

#define EPSZERO 1E-12

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif


class Vector
{
public:

  HOST DEVICE inline Vector(const double cx=0, const double cy=0, const double cz=0)
    {
	coord[0] = cx;
	coord[1] = cy;
        coord[2] = cz;
    }

  HOST DEVICE  inline Vector(const Vector &v)
    {
	coord[0] = v.coord[0];
	coord[1] = v.coord[1];
	coord[2] = v.coord[2];
    }

  HOST DEVICE  inline Vector &operator=(const Vector &v)
    {
	coord[0] = v.coord[0];
	coord[1] = v.coord[1];
	coord[2] = v.coord[2];

	return *this;
    }

    
  HOST DEVICE  inline double operator,(const Vector &v1) const
    {
	return
	    (v1.coord[0] * coord[0] +
	     v1.coord[1] * coord[1] +
	     v1.coord[2] * coord[2]);
    }

  HOST DEVICE  inline Vector operator*(const Vector &v1) const
    {
	return
	    Vector (coord[1] * v1.coord[2] -
		    v1.coord[1] * coord[2],
		    coord[2] * v1.coord[0] -
		    v1.coord[2] * coord[0],
		    coord[0] * v1.coord[1] -
		    v1.coord[0] * coord[1]);
    }

  HOST DEVICE  inline Vector operator*(const double c) const
    {
	return
	    Vector (coord[0]*c,
		    coord[1]*c,
		    coord[2]*c);
    }

  HOST DEVICE  inline Vector &operator+=(const Vector &v1)
    {
	coord[0] += v1.coord[0];
	coord[1] += v1.coord[1];
	coord[2] += v1.coord[2];

	return *this;
    }

  HOST DEVICE   inline Vector &operator-=(const Vector &v1)
    {
	coord[0] -= v1.coord[0];
	coord[1] -= v1.coord[1];
	coord[2] -= v1.coord[2];

	return *this;
    }
    
  HOST DEVICE  inline Vector &operator*=(const double c)
    {
	coord[0] *= c;
	coord[1] *= c;
	coord[2] *= c;

	return *this;
    }

  HOST DEVICE inline Vector &operator/=(double c)
    {
	c = 1./c;
	
	coord[0] *= c;
	coord[1] *= c;
	coord[2] *= c;

	return *this;
    }
    
  HOST DEVICE inline Vector operator+(const Vector &v1) const
    {
	return Vector(coord[0] + v1.coord[0],
		      coord[1] + v1.coord[1],
		      coord[2] + v1.coord[2]);
    }

  HOST DEVICE inline Vector operator-(const Vector &v1) const
    {

	return Vector(coord[0] - v1.coord[0],
		      coord[1] - v1.coord[1],
		      coord[2] - v1.coord[2]);
    }

  HOST DEVICE inline Vector operator-() const
    {

	return Vector(-coord[0],
		      -coord[1],
		      -coord[2]);
    }

  HOST DEVICE inline Vector operator/(double c) const
    {
	c = 1./c;

	return Vector(c*coord[0],
		      c*coord[1],
		      c*coord[2]);
    }

  HOST DEVICE inline double sizesqd() const
    {
	return
	    coord[0]*coord[0] +
	    coord[1]*coord[1] +
	    coord[2]*coord[2];
    }

  HOST DEVICE inline double size() const
    {
	return
	    sqrt(sizesqd());
    }
    
  HOST DEVICE inline Vector unit() const
    {
	return (*this)/size();
    }
    
  HOST DEVICE void Print() {
	//printf("(%lf,%lf,%lf)\n",coord[0],coord[1],coord[2]);
    }

  HOST DEVICE inline int iszero() const
  {
    if (sizesqd() < EPSZERO) 
      return 1;
    return 0;
  }
    
    double coord[3];
    
};

//static Vector temp;

// More efficient (but less elegant) versions of some of the
// operators above.

/*inline Vector &Mult(Vector &vin, double c, Vector &vout)
{
    vout.coord[0] = vin.coord[0]*c;
    vout.coord[1] = vin.coord[1]*c;
    vout.coord[2] = vin.coord[2]*c;
    
    return vout;
}

inline Vector &Add(Vector &v1, Vector &v2, Vector &vout)
{
    vout.coord[0] = v1.coord[0] + v2.coord[0];
    vout.coord[1] = v1.coord[1] + v2.coord[1];
    vout.coord[2] = v1.coord[2] + v2.coord[2];

    return vout;
}

inline Vector &Sub(Vector &v1, Vector&v2, Vector &vout)
{
    vout.coord[0] = v1.coord[0] - v2.coord[0];
    vout.coord[1] = v1.coord[1] - v2.coord[1];
    vout.coord[2] = v1.coord[2] - v2.coord[2];

    return vout;
}

inline Vector &Negate(Vector &vin, Vector &vout)
{

    vout.coord[0] = -vin.coord[0];
    vout.coord[1] = -vin.coord[1];
    vout.coord[2] = -vin.coord[2];

    return vout;
}

inline Vector &Div(Vector &vin, double c, Vector &vout)
{
    c = 1./c;

    vout.coord[0] = c*vin.coord[0];
    vout.coord[1] = c*vin.coord[1];
    vout.coord[2] = c*vin.coord[2];

    return vout;
}

inline Vector &Cross(Vector &v1, Vector &v2, Vector &vout)
{
    vout.coord[0] = v1.coord[1] * v2.coord[2] - v2.coord[1] * v1.coord[2];
    vout.coord[1] = v1.coord[2] * v2.coord[0] - v2.coord[2] * v1.coord[0];
    vout.coord[2] = v1.coord[0] * v2.coord[1] - v2.coord[0] * v1.coord[1];

    return vout;
}

inline Vector &Unit(Vector &vin, Vector &vout)
{
    double size = vin.size();
    
    vout.coord[0] = vin.coord[0]/size;
    vout.coord[1] = vin.coord[1]/size;
    vout.coord[2] = vin.coord[2]/size;

    return vout;
}
*/

inline double distance(const Vector &v1, const Vector &v2)
{
    return (v1-v2).size();
}

inline double dsqrd(const Vector &v1, const Vector &v2)
{
    return (v1-v2).sizesqd();
}

#endif
