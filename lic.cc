#include "point.h"
#include "vector.h"
#include "stdlib.h"


double maxvecmag=0.;

void licGPU(int rows, int cols, Vector *vecdata, int *texdata, double *IdataGPU);

class ColorTable {
public:
  Point *ct;
  int num;

  ColorTable(int i) {
    num = i;
    ct = new Point[i];
  }

  inline Point operator[](double key) {
    Point color;
    int i = (int)(key*num);

    color = ct[i]*(1-key*num+i)+ct[i+1]*(key*num-i);
    //printf("key: %lf i: %d (key*num-i): %lf (1-key*num+i): %lf\n",key,i,(key*num-i),(1-key*num+i));
    return color;
  }

  inline Point &operator[](int i) {
    return ct[i];
  }
};


ColorTable table(NUMCOLORS);

class StreamLine {
public:
  Point fwd[M+L-1];
  Point bwd[M+L-1];
  Point origin;

  void GenStreamLine(int, int);

  inline Point &operator[](int m) {
    if (m == 0)
      return origin;
    else if (m>0)
      return fwd[m-1];
    else
      return bwd[-m-1];

  }

private:
  inline void RK(Point &, double);

};

class RegGrid {
public:
  int rows;
  int cols;
  Vector **vecdata;
  Vector *vecdata_1d;
  int **hitdata;
  int **texdata;
  int *texdata_1d;
  double **Idata;
  int numvalid;

  RegGrid(int r, int c)
  {
    cols = c;
    rows = r;

    vecdata = new Vector *[r];
    vecdata_1d = new Vector [r*c];
    hitdata = new int *[r];
    texdata = new int *[r];
    texdata_1d = new int [r*c];
    Idata = new double *[r];
    numvalid=0;

    for (int i=0; i<r; i++) {
      vecdata[i] = new Vector[c];
      hitdata[i] = new int[c];
      texdata[i] = new int[c];
      Idata[i] = new double[c];
      for (int j=0; j<c; j++) {
      	hitdata[i][j] = 0;
      	Idata[i][j] = 0;
      	texdata[i][j] = 0;
      }
    }
  }

  inline int getPixel(const Point &p, int &i, int &j)
  {
    i = ((int)p.coord[0] + rows)%rows;
    j = ((int)p.coord[1] + cols)%cols;

    if (i >= 0 && i < rows && j >=0 && j < cols)
      return 1;
    return 0;
  }

  inline Point getCenter(int i, int j)
  {
    return Point(i+.5,j+.5);
  }

  inline double ComputeI(StreamLine &s, int &numvalid)
  {
    double T,k,I;
    int i,j;

    T=0;
    numvalid = 0;

    for(i=-L; i<= L; i++) {
      if (validpt(s[i])) {
        T += getT(s[i]);
        numvalid++;
      }
    }
    if (getPixel(s[0],i,j)) {
      k = 1./numvalid;
      Idata[i][j] += I = T*k;
      hitdata[i][j]++;
      return I;
    }
    return 0;

  }

  inline double ComputeIFwd(StreamLine &s, double &I, int m, int &numvalid)
  {
    int i,j;
    double k;

    if (getPixel(s[m],i,j)) {
      if (validpt(s[m+L]))
        numvalid++;
      if (validpt(s[m-1-L]))
        numvalid--;
      k = 1./numvalid;
      Idata[i][j] += I += k*(getT(s[m+L]) - getT(s[m-1-L]));
      hitdata[i][j]++;
      return I;
    }
    return 0;
  }

  inline double ComputeIBwd(StreamLine &s, double &I, int m, int &numvalid)
  {
    int i,j;
    double k;

    if (getPixel(s[m],i,j)) {
      if (validpt(s[m-L]))
        numvalid++;
      if (validpt(s[m+1+L]))
        numvalid--;
      k = 1./numvalid;
      Idata[i][j] += I += k*(getT(s[m-L]) - getT(s[m+1+L]));
      hitdata[i][j]++;
      return I;
    }
    return 0;
  }

  inline double getT(Point &p)
  {
    int i,j;

    if (getPixel(p,i,j))
      return texdata[i][j];
    return 0;
  }

  inline Vector getVector(const Point &p)
  {
    int i,j;

    if (getPixel(p,i,j)) {

      return vecdata[i][j];
    }
    return(Vector(0,0,0));
  }

  inline void Normalize()
  {
    for (int i=0; i<rows; i++)
      for (int j=0; j<cols; j++) {
        Idata[i][j] /= hitdata[i][j];
      }
  }

  inline int validpt(Point &p) {
    int i,j;

    if (getPixel(p,i,j))
      return 1;
    return 0;
  }

  void Print()
  {
    double scale;
    double magind;
    double mag;
    Point color;

    //printf("P2\t%d\t%d\t255\n",rows,cols);
    printf("P3\t%d\t%d\t255\n",cols,rows);
    for (int i=0; i<rows; i++)
      for (int j=0; j<cols; j++) {
        //printf("%d ",(int)Idata[i][j]);
        scale = Idata[i][j]/255.;
        mag = vecdata[i][j].size();
        magind = mag/maxvecmag;
        //printf("magind:%lf\n",magind);
        color = table[magind]*scale;
        //	color.Print();
        printf("%d %d %d ",(int)color.coord[0],(int)color.coord[1],
               (int)color.coord[2]);
      }
  }

  void Print2(double *IdataGPU)
  {
    double scale;
    double magind;
    double mag;
    Point color;

    //printf("P2\t%d\t%d\t255\n",rows,cols);
    printf("P3\t%d\t%d\t255\n",cols,rows);
    for (int i=0; i<rows; i++)
      for (int j=0; j<cols; j++) {
        //printf("%d ",(int)Idata[i][j]);
        scale = IdataGPU[i*cols+j]/255.;
        mag = vecdata[i][j].size();
        magind = mag/maxvecmag;
        //printf("magind:%lf\n",magind);
        color = table[magind]*scale;
        //  color.Print();
        printf("%d %d %d ",(int)color.coord[0],(int)color.coord[1],
               (int)color.coord[2]);
      }
  }

};

RegGrid *grid;

void StreamLine::GenStreamLine(int i, int j)
{
  Point b,f;

  origin = f = b = grid->getCenter(i,j);
  for (int k=0; k<M+L-1; k++) {
    RK(f,Ht);
    fwd[k] = f;
    RK(b,-Ht);
    bwd[k] = b;
  }
}

inline void StreamLine::RK(Point &p, double h)
{
  Vector v;
  Vector k1,k2,k3,k4;

  v = grid->getVector(p);
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k1 = v*h;
  v = (grid->getVector(p+k1*.5));
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k2 = v*h;
  v = (grid->getVector(p+k2*.5));
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k3 = v*h;
  v = (grid->getVector(p+k3));
  if (!v.iszero())
    v = v.unit();
  //v.Print();

  k4 = v*h;
  p += k1/6 + k2/3 + k3/3 + k4/6;

  return;
}

void
readPts(char *vecfname, char *texfname, RegGrid *grid)
{
  double vx,vy;
  int t;
  int dummy;
  double mag;

  FILE *vfp = fopen( vecfname,  "r");
  FILE *tfp = fopen( texfname,  "r");

  // if (!vfp) {
  //   printf("Bad Vector Filename: %s\n",vfp);
  //   exit(-1);
  // }

  // if (!tfp) {
  //   printf("Bad Texture Filename: %s\n",tfp);
  //   exit(-1);
  // }


  for (int i=0; i<grid->rows; i++) {
    for (int j=0; j<grid->cols; j++) {
      dummy = fscanf(vfp,"%lf %lf\n",&vx,&vy);
      grid->vecdata[i][j].coord[0] = vx;
      grid->vecdata[i][j].coord[1] = vy;

      grid->vecdata_1d[i*grid->cols + j].coord[0] = vx;
      grid->vecdata_1d[i*grid->cols + j].coord[1] = vy;

      mag = sqrt(vx*vx+vy*vy);
      if (mag > maxvecmag)
        maxvecmag = mag;

      dummy = fscanf(tfp,"%d",&t);
      grid->texdata[i][j] = t;
      grid->texdata_1d[i*grid->cols + j] = t;
    }
  }
  fclose(vfp);
  fclose(tfp);
}

void
lic()
{
  double I0;
  double I;
  int m;
  StreamLine s;
  int nsum=0, tmpsum=0;

  for (int i=0; i<grid->rows; i++) {
    for (int j=0; j<grid->cols; j++) {
      if (grid->hitdata[i][j] < MINNUMHITS) {
        s.GenStreamLine(i,j);
        I = I0 = grid->ComputeI(s,nsum);
        tmpsum = nsum;
        for (m=1; m < M; m++)
          grid->ComputeIFwd(s,I,m,tmpsum);
        I = I0;
        tmpsum = nsum;
        for (m=1; m < M; m++)
          grid->ComputeIBwd(s,I,-m,tmpsum);
      }
    }
  }
  grid->Normalize();
}

void
docolors() {
  table[0] = Point(100,100,100);
  table[1] = Point(150,0,0);
  table[2] = Point(255,0,0);
  table[3] = Point(0,0,255);
}


main(int argc, char *argv[])
{
  int rows, cols;
  int cpu_or_gpu = 0;

  if (argc < 5) {
    printf("Usage: %s [vec file] [tex file] [rows] [cols] [gpu?:0/1]\n",argv[0]);
    exit(-1);
  }

  rows = atoi(argv[3]);
  cols = atoi(argv[4]);
  if(argc == 6)
    cpu_or_gpu =  atoi(argv[5]);

  grid = new RegGrid(rows,cols);
  readPts(argv[1],argv[2],grid);
  docolors();
  if(cpu_or_gpu == 0) {
    lic();
    grid->Print();
  } else {
    double *IdataGPU;
    IdataGPU = new double [rows * cols];
    licGPU(grid->rows, grid->cols, grid->vecdata_1d, grid->texdata_1d, IdataGPU);
    grid->Print2(IdataGPU);
  }
}

