#ifndef linearAlgebraOperations_h
#define linearAlgebraOperations_h

extern "C"
{
  void
  dgemv_(const char *TRANS,
         const unsigned int *M,
         const unsigned int *N,
         const double *alpha,
         const double *A,
         const unsigned int *LDA,
         const double *X,
         const unsigned int *INCX,
         const double *beta,
         double *C,
         const unsigned int *INCY);

  void
  sgemv_(const char *TRANS,
         const unsigned int *M,
         const unsigned int *N,
         const float *alpha,
         const float *A,
         const unsigned int *LDA,
         const float *X,
         const unsigned int *INCX,
         const float *beta,
         float *C,
         const unsigned int *INCY);

  void
  dsymv_(const char *UPLO,
         const unsigned int *N,
         const double *alpha,
         const double *A,
         const unsigned int *LDA,
         const double *X,
         const unsigned int *INCX,
         const double *beta,
         double *C,
         const unsigned int *INCY);
  void
  dgesv_(int *n,
         int *nrhs,
         double *a,
         int *lda,
         int *ipiv,
         double *b,
         int *ldb,
         int *info);
  void
  dsysv_(const char *UPLO,
         const int *n,
         const int *nrhs,
         double *a,
         const int *lda,
         int *ipiv,
         double *b,
         const int *ldb,
         double *work,
         const int *lwork,
         int *info);
  void
  dscal_(const unsigned int *n,
         const double *alpha,
         double *x,
         const unsigned int *inc);
  void
  sscal_(const unsigned int *n,
         const float *alpha,
         float *x,
         const unsigned int *inc);
  void
  daxpy_(const unsigned int *n,
         const double *alpha,
         const double *x,
         const unsigned int *incx,
         double *y,
         const unsigned int *incy);
  void
  saxpy_(const unsigned int *n,
         const float *alpha,
         const float *x,
         const unsigned int *incx,
         float *y,
         const unsigned int *incy);
  void
  dgemm_(const char *transA,
         const char *transB,
         const unsigned int *m,
         const unsigned int *n,
         const unsigned int *k,
         const double *alpha,
         const double *A,
         const unsigned int *lda,
         const double *B,
         const unsigned int *ldb,
         const double *beta,
         double *C,
         const unsigned int *ldc);
  void
  sgemm_(const char *transA,
         const char *transB,
         const unsigned int *m,
         const unsigned int *n,
         const unsigned int *k,
         const float *alpha,
         const float *A,
         const unsigned int *lda,
         const float *B,
         const unsigned int *ldb,
         const float *beta,
         float *C,
         const unsigned int *ldc);
  void
  dsyevd_(const char *jobz,
          const char *uplo,
          const unsigned int *n,
          double *A,
          const unsigned int *lda,
          double *w,
          double *work,
          const unsigned int *lwork,
          int *iwork,
          const unsigned int *liwork,
          int *info);
  void
  dsygvx_(const int *itype,
          const char *jobz,
          const char *range,
          const char *uplo,
          const int *n,
          double *a,
          const int *lda,
          double *b,
          const int *ldb,
          const double *vl,
          const double *vu,
          const int *il,
          const int *iu,
          const double *abstol,
          int *m,
          double *w,
          double *z,
          const int *ldz,
          double *work,
          const int *lwork,
          int *iwork,
          int *ifail,
          int *info);
  void
  dsyevx_(const char *jobz,
          const char *range,
          const char *uplo,
          const int *n,
          double *a,
          const int *lda,
          const double *vl,
          const double *vu,
          const int *il,
          const int *iu,
          const double *abstol,
          int *m,
          double *w,
          double *z,
          const int *ldz,
          double *work,
          const int *lwork,
          int *iwork,
          int *ifail,
          int *info);
  double
  dlamch_(const char *cmach);
  void
  dsyevr_(const char *jobz,
          const char *range,
          const char *uplo,
          const unsigned int *n,
          double *A,
          const unsigned int *lda,
          const double *vl,
          const double *vu,
          const unsigned int *il,
          const unsigned int *iu,
          const double *abstol,
          const unsigned int *m,
          double *w,
          double *Z,
          const unsigned int *ldz,
          unsigned int *isuppz,
          double *work,
          const int *lwork,
          int *iwork,
          const int *liwork,
          int *info);
  void
  dsyrk_(const char *uplo,
         const char *trans,
         const unsigned int *n,
         const unsigned int *k,
         const double *alpha,
         const double *A,
         const unsigned int *lda,
         const double *beta,
         double *C,
         const unsigned int *ldc);
  void
  dsyr_(const char *uplo,
        const unsigned int *n,
        const double *alpha,
        const double *X,
        const unsigned int *incx,
        double *A,
        const unsigned int *lda);
  void
  dsyr2_(const char *uplo,
         const unsigned int *n,
         const double *alpha,
         const double *x,
         const unsigned int *incx,
         const double *y,
         const unsigned int *incy,
         double *a,
         const unsigned int *lda);
  void
  dcopy_(const unsigned int *n,
         const double *x,
         const unsigned int *incx,
         double *y,
         const unsigned int *incy);
  void
  scopy_(const unsigned int *n,
         const float *x,
         const unsigned int *incx,
         float *y,
         const unsigned int *incy);
  double
  ddot_(const unsigned int *N,
        const double *X,
        const unsigned int *INCX,
        const double *Y,
        const unsigned int *INCY);

  double
  dnrm2_(const unsigned int *n, const double *x, const unsigned int *incx);

  void
  dpotrf_(const char *uplo,
          const unsigned int *n,
          double *a,
          const unsigned int *lda,
          int *info);
  void
  dpotri_(const char *uplo,
          const unsigned int *n,
          double *A,
          const unsigned int *lda,
          int *info);

  void
  dtrtri_(const char *uplo,
          const char *diag,
          const unsigned int *n,
          double *a,
          const unsigned int *lda,
          int *info);

  // LU decomoposition of a general matrix
  void
  dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

  // generate inverse of a matrix given its LU decomposition
  void
  dgetri_(int *N,
          double *A,
          int *lda,
          int *IPIV,
          double *WORK,
          int *lwork,
          int *INFO);
}
#endif
