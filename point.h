#ifndef POINT_H
#define POINT_H

#include "vector.h"

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#define MINNUMHITS 12
#define Ht .5
#define M 10
#define L 10
#define NUMCOLORS 4

class Point
{

public:

  HOST DEVICE inline Point(const double cx=0, const double cy=0, const double
		 cz=0, const double ch=1)
    {
	coord[0] = cx;
	coord[1] = cy;
	coord[2] = cz;
	coord[3] = ch;
    }

  HOST DEVICE inline Point(const Point &p)
    {
	coord[0] = p.coord[0];
	coord[1] = p.coord[1];
	coord[2] = p.coord[2];
	coord[3] = p.coord[3];
    }

  HOST DEVICE inline Vector operator-(const Point &p) const
    {
	return Vector(coord[0]-p.coord[0],
		      coord[1]-p.coord[1],
		      coord[2]-p.coord[2]
		      );
    }

  HOST DEVICE inline Vector operator-(const Vector &v) const
    {
	return Vector(coord[0]-v.coord[0],
		      coord[1]-v.coord[1],
		      coord[2]-v.coord[2]);
    }

  HOST DEVICE inline Point operator+(const Vector &v) const
    {
	return Point(coord[0]+v.coord[0],
		     coord[1]+v.coord[1],
		     coord[2]+v.coord[2],
		     1.);
    }

  HOST DEVICE inline Point operator+(const Point &p) const
    {
	return Point(coord[0]+p.coord[0],
		     coord[1]+p.coord[1],
		     coord[2]+p.coord[2],
		     1.);
    }

  HOST DEVICE inline Point &operator+=(const Point &p)
    {
	coord[0] += p.coord[0];
	coord[1] += p.coord[1];
	coord[2] += p.coord[2];
	coord[3] = 1.;

	return (*this);
    }

  HOST DEVICE inline Point &operator+=(const Vector &v)
    {
	coord[0] += v.coord[0];
	coord[1] += v.coord[1];
	coord[2] += v.coord[2];
	coord[3] = 1.;

	return (*this);
    }

  HOST DEVICE inline Point &operator-=(const Point &p)
    {
	coord[0] -= p.coord[0];
	coord[1] -= p.coord[1];
	coord[2] -= p.coord[2];
	coord[3] = 1.;

	return *this;
    }

  HOST DEVICE inline Point ToBasis(const Point &c, const Vector &u,
			 const Vector &v, const Vector &w) const
    {
	Vector v1((*this)-c);

	return Point((v1,u),(v1,v),(v1,w));
    }

  HOST DEVICE inline Point ToCanonical(const Point &c, const Vector &u,
			     const Vector &v, const Vector &w) const
    {
	return Point(coord[0]*u.coord[0]+
		     coord[1]*v.coord[0]+
		     coord[2]*w.coord[0],
		     coord[0]*u.coord[1]+
		     coord[1]*v.coord[1]+
		     coord[2]*w.coord[1],
		     coord[0]*u.coord[2]+
		     coord[1]*v.coord[2]+
		     coord[2]*w.coord[2]) + c;
    }

  HOST DEVICE inline Point &operator=(const Vector &v)
    {
	coord[0] = v.coord[0];
	coord[1] = v.coord[1];
	coord[2] = v.coord[2];
	coord[3] = 1.;

	return *this;
    }

  HOST DEVICE inline Point &operator=(const Point &p)
    {
	coord[0] = p.coord[0];
	coord[1] = p.coord[1];
	coord[2] = p.coord[2];
	coord[3] = p.coord[3];

	return *this;
    }

  HOST DEVICE inline Vector operator*(const Vector &v2) const
    {
	Vector v1 = Vector(coord[0],coord[1],coord[2]);
	return v1*v2;
    }

  HOST DEVICE inline Point operator*(const double c) const
    {
	return Point(coord[0]*c,coord[1]*c,coord[2]*c,coord[3]);
    }

  HOST DEVICE inline Point operator/(double c) const
    {
	double div = 1./c;
	return (*this)*div;
    }

  HOST DEVICE inline double operator,(const Vector &v) const
    {
	return
	    coord[0]*v.coord[0] +
	    coord[1]*v.coord[1] +
	    coord[2]*v.coord[2];
    }

  HOST DEVICE inline double operator,(const Point &p) const
    {
	return
	    coord[0]*p.coord[0] +
	    coord[1]*p.coord[1] +
	    coord[2]*p.coord[2];
    }

  HOST DEVICE void Print() {
	printf("%lf %lf %lf %lf\n",coord[0],coord[1],coord[2],coord[3]);
    }

  HOST DEVICE inline operator Vector() const
    {
	return (Vector(coord[0],coord[1],coord[2]));
    }

    double coord[4];

};

inline void blend(const Point &p1, const Point &p2,
		  const double t, Point &pr)
{
    double w1,w2;

    w1 = p1.coord[3]*(1.-t);
    w2 = p2.coord[3]*t;

    pr.coord[3] = w1+w2;

    w1 /= pr.coord[3];
    w2 = 1.-w1;

    pr.coord[0] = w1*p1.coord[0] + w2*p2.coord[0];
    pr.coord[1] = w1*p1.coord[1] + w2*p2.coord[1];
    pr.coord[2] = w1*p1.coord[2] + w2*p2.coord[2];
}


inline double distance(const Point &p1, const Point &p2)
{
    return
	sqrt((p1.coord[0]-p2.coord[0])*(p1.coord[0]-p2.coord[0]) +
	     (p1.coord[1]-p2.coord[1])*(p1.coord[1]-p2.coord[1]) +
	     (p1.coord[2]-p2.coord[2])*(p1.coord[2]-p2.coord[2]));
}

#endif

