#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

inline int signd(const double x) { return (x >= 0 ? 1 : -1); }

double oppositeOfMinor(const Eigen::Matrix3d &M, const int row, const int col) {
  int x1 = col == 0 ? 1 : 0;
  int x2 = col == 2 ? 1 : 2;
  int y1 = row == 0 ? 1 : 0;
  int y2 = row == 2 ? 1 : 2;

  return (M(y1, x2) * M(y2, x1) - M(y1, x1) * M(y2, x2));
}

// computes R = H( I - (2/v)*te_star*ne_t )
void findRmatFrom_tstar_n(const Eigen::Vector3d &tstar,
                          const Eigen::Matrix3d &H, const Eigen::Vector3d &n,
                          const double v, Eigen::Matrix3d &R) {
  Eigen::Vector3d tstar_m = tstar;
  Eigen::Vector3d n_m = n;
  Eigen::Matrix3d I;
  I << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  R = H * (I - (2.0 / v) * tstar_m * n_m.transpose());
  if (R.determinant() < 0) {
    R = -R;
  }
}

void hdecom_malis(Eigen::Matrix3d &HH, std::vector<Eigen::Matrix3d> &Rest,
                  std::vector<Eigen::Vector3d> &Test,
                  std::vector<Eigen::Vector3d> &Nest) {

  Eigen::Matrix3d H = HH;
  Eigen::Matrix3d M = H.transpose() * H;
  double m11 = M(0, 0);
  double m12 = M(0, 1);
  double m13 = M(0, 2);
  double m22 = M(1, 1);
  double m23 = M(1, 2);
  double m33 = M(2, 2);

  double c2 = -(m11 + m22 + m33);
  double c1 =
      m11 * m22 + m11 * m33 + m22 * m33 - (m12 * m12 + m13 * m13 + m23 * m23);
  double c0 = m12 * m12 * m33 + m13 * m13 * m22 + m23 * m23 * m11 -
              m11 * m22 * m33 - 2 * m12 * m13 * m23;

  double a = c1 - c2 * c2 / 3.0;
  double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
  double c = 3.0 * b / (2.0 * a) * std::sqrt((-3.0 / a));

  if (abs(c) < 0.99999) {
    double d = 2.0 * std::sqrt((-a / 3.0));
    // double sigma =
    //     d * cos(acos(c) / 3.0 - 2.09439510239319526263557236234192) - c2
    //     / 3.0;
    double sigma = d * sin(asin(-c) / 3.0) - c2 / 3.0;
    H = H / sqrt((sigma));

    Eigen::Matrix3d S;
    S = H.transpose() * H;
    S(0, 0) -= 1.0;
    S(1, 1) -= 1.0;
    S(2, 2) -= 1.0;

    if (S.lpNorm<Eigen::Infinity>() < 0.001) {
      Rest.push_back(Eigen::Matrix3d::Identity());
      Test.push_back(Eigen::Vector3d(1.0, 1.0, 1.0));
      return;
    }

    //! Compute nvectors
    Eigen::Vector3d npa, npb;

    double M00 = oppositeOfMinor(S, 0, 0);
    double M11 = oppositeOfMinor(S, 1, 1);
    double M22 = oppositeOfMinor(S, 2, 2);

    double rtM00 = sqrt(M00);
    double rtM11 = sqrt(M11);
    double rtM22 = sqrt(M22);

    double M01 = oppositeOfMinor(S, 0, 1);
    double M12 = oppositeOfMinor(S, 1, 2);
    double M02 = oppositeOfMinor(S, 0, 2);

    int e12 = signd(M12);
    int e02 = signd(M02);
    int e01 = signd(M01);

    double nS00 = abs(S(0, 0));
    double nS11 = abs(S(1, 1));
    double nS22 = abs(S(2, 2));

    // find max( |Sii| ), i=0, 1, 2
    int indx = 0;
    if (nS00 < nS11) {
      indx = 1;
      if (nS11 < nS22)
        indx = 2;
    } else {
      if (nS00 < nS22)
        indx = 2;
    }

    switch (indx) {
    case 0:
      npa[0] = S(0, 0), npb[0] = S(0, 0);
      npa[1] = S(0, 1) + rtM22, npb[1] = S(0, 1) - rtM22;
      npa[2] = S(0, 2) + e12 * rtM11, npb[2] = S(0, 2) - e12 * rtM11;
      break;
    case 1:
      npa[0] = S(0, 1) + rtM22, npb[0] = S(0, 1) - rtM22;
      npa[1] = S(1, 1), npb[1] = S(1, 1);
      npa[2] = S(1, 2) - e02 * rtM00, npb[2] = S(1, 2) + e02 * rtM00;
      break;
    case 2:
      npa[0] = S(0, 2) + e01 * rtM11, npb[0] = S(0, 2) - e01 * rtM11;
      npa[1] = S(1, 2) + rtM00, npb[1] = S(1, 2) - rtM00;
      npa[2] = S(2, 2), npb[2] = S(2, 2);
      break;
    default:
      break;
    }

    double traceS = S(0, 0) + S(1, 1) + S(2, 2);
    double v = 2.0 * sqrt(1.0 + traceS - M00 - M11 - M22);

    double ESii = signd(S(indx, indx));
    double r_2 = 2.0 + traceS + v;
    double nt_2 = 2.0 + traceS - v;

    double r = sqrt(r_2);
    double n_t = sqrt(nt_2);

    Eigen::Vector3d na = npa / npa.norm();
    Eigen::Vector3d nb = npb / npb.norm();

    double half_nt = 0.5 * n_t;
    double esii_t_r = ESii * r;

    Eigen::Vector3d ta_star = half_nt * (esii_t_r * nb - n_t * na);
    Eigen::Vector3d tb_star = half_nt * (esii_t_r * na - n_t * nb);

    Eigen::Matrix3d Ra, Rb;
    Eigen::Vector3d ta, tb;

    // Ra, ta, na
    findRmatFrom_tstar_n(ta_star, H, na, v, Ra);
    ta = Ra * ta_star;

    Rest.push_back(Ra);
    Test.push_back(ta);
    Nest.push_back(na);

    // Rb, tb, nb
    findRmatFrom_tstar_n(tb_star, H, nb, v, Rb);
    tb = Rb * tb_star;

    Rest.push_back(Rb);
    Test.push_back(tb);
    Nest.push_back(na);

  } else {
    hdecom_svd(HH, Rest, Test, Nest);
  }
}