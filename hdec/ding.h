// Author: Yaqing Ding (yaqing.ding@cvut.cz)
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

inline void refine_lambda_s(double &s, double &p, double &q, const double h1,
                            const double h2, const double h3, const double h4,
                            const double h5, const double h6, const double h7,
                            const double h8, const double h9, const double h11,
                            const double h22, const double h33,
                            const double h44, const double h55,
                            const double h66, const double h77,
                            const double h88, const double h99, const double g1,
                            const double g2, const double g3) {
  for (int iter = 0; iter < 5; ++iter) {
    double psqr = p * p;
    double qsqr = q * q;
    double g4 = h11 + h44 + h77 - s;
    double g5 = h33 + h66 + h99 - s;
    double g6 = h22 + h55 + h88 - s;
    double r1 = g4 * psqr - 2.0 * p * g1 + g5;
    double r2 = g6 * qsqr - 2.0 * q * g3 + g5;
    double r3 = -p * g1 - q * g3 + g5 + g2 * p * q;

    if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-12)
      return;
    double J11 = -psqr - 1.0;
    double J12 = 2.0 * (p * g4 - g1);
    double J21 = -qsqr - 1.0;
    double J23 = 2.0 * (q * g6 - g3);

    double J32 = g2 * q - g1;
    double J33 = g2 * p - g3;
    double detJ = 1.0 / (-J12 * J23 - J11 * J23 * J32 - J12 * J21 * J33);
    s += (J23 * J32 * r1 + J12 * J33 * r2 - J12 * J23 * r3) * detJ;
    p += ((J23 + J21 * J33) * r1 - J11 * J33 * r2 + J11 * J23 * r3) * detJ;
    q += (-J21 * J32 * r1 + (J12 + J11 * J32) * r2 + J12 * J21 * r3) * detJ;
  }
}

inline void refine_lambda3(double &p, double &q, const double h1,
                           const double h2, const double h3, const double h4,
                           const double h5, const double h6, const double h7,
                           const double h8, const double h9) {

  for (int iter = 0; iter < 5; ++iter) {
    double r1 = h1 * h1 * p * p - 2.0 * h1 * h3 * p + h3 * h3 +
                h4 * h4 * p * p - 2.0 * h4 * h6 * p + h6 * h6 +
                h7 * h7 * p * p - 2.0 * h7 * h9 * p + h9 * h9 - p * p - 1.0;
    double r2 = h2 * h2 * q * q - 2.0 * h2 * h3 * q + h3 * h3 +
                h5 * h5 * q * q - 2.0 * h5 * h6 * q + h6 * h6 +
                h8 * h8 * q * q - 2.0 * h8 * h9 * q + h9 * h9 - q * q - 1.0;
    double r3 = h1 * h1 * p * p - 2 * h1 * h2 * p * q + h2 * h2 * q * q +
                h4 * h4 * p * p - 2 * h4 * h5 * p * q + h5 * h5 * q * q +
                h7 * h7 * p * p - 2 * h7 * h8 * p * q + h8 * h8 * q * q -
                p * p - q * q;
    if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-12)
      return;
    double J11 =
        -p - h1 * (h3 - h1 * p) - h4 * (h6 - h4 * p) - h7 * (h9 - h7 * p);
    double J22 =
        -q - h2 * (h3 - h2 * q) - h5 * (h6 - h5 * q) - h8 * (h9 - h8 * q);
    double J31 = h1 * (h1 * p - h2 * q) - p + h4 * (h4 * p - h5 * q) +
                 h7 * (h7 * p - h8 * q);
    double J32 = -q - h2 * (h1 * p - h2 * q) - h5 * (h4 * p - h5 * q) -
                 h8 * (h7 * p - h8 * q);
    double detJ =
        0.5 / (J11 * J11 * J22 * J22 + J11 * J11 * J32 * J32 +
               J22 * J22 * J31 * J31);  
    p -= (J11 * r1 * (J22 * J22 + J32 * J32) -
          r3 * (J31 * J32 * J32 - J31 * (J22 * J22 + J32 * J32)) -
          J22 * J31 * J32 * r2) *
         detJ;
    q -= (J22 * r2 * (J11 * J11 + J31 * J31) -
          r3 * (J31 * J31 * J32 - J32 * (J11 * J11 + J31 * J31)) -
          J11 * J31 * J32 * r1) *
         detJ;
  }
}

inline void refine_lambda(double &p, double &q, const double h1,
                          const double h2, const double h3, const double h4,
                          const double h5, const double h6, const double h7,
                          const double h8, const double h9) {

  for (int iter = 0; iter < 5; ++iter) {
    double r1 = h1 * h1 * p * p - 2.0 * h1 * h3 * p + h3 * h3 +
                h4 * h4 * p * p - 2.0 * h4 * h6 * p + h6 * h6 +
                h7 * h7 * p * p - 2.0 * h7 * h9 * p + h9 * h9 - p * p - 1.0;
    double r2 = h2 * h2 * q * q - 2.0 * h2 * h3 * q + h3 * h3 +
                h5 * h5 * q * q - 2.0 * h5 * h6 * q + h6 * h6 +
                h8 * h8 * q * q - 2.0 * h8 * h9 * q + h9 * h9 - q * q - 1.0;
    if (std::abs(r1) + std::abs(r2) < 1e-12)
      return;
    double J11 =
        -p - h1 * (h3 - h1 * p) - h4 * (h6 - h4 * p) - h7 * (h9 - h7 * p);
    double J22 =
        -q - h2 * (h3 - h2 * q) - h5 * (h6 - h5 * q) - h8 * (h9 - h8 * q);
    double detJ = 0.5 / (J11 * J22);
    p += (-J22 * r1) * detJ;
    q += (-J11 * r2) * detJ;
  }
}

void solve_quadratic(double a, double b, double c, double roots[2]) {

  double sq = std::sqrt(b * b - 4.0 * a * c);

  roots[0] = (b > 0) ? (2.0 * c) / (-b - sq) : (2.0 * c) / (-b + sq);
  roots[1] = c / (a * roots[0]);
}

void hdecom_ding(Eigen::Matrix3d &HH, std::vector<Eigen::Matrix3d> &Rest,
                 std::vector<Eigen::Vector3d> &Test,
                 std::vector<Eigen::Vector3d> &Nest) {

  Eigen::Matrix3d H = HH;
  Eigen::Matrix3d M = H.transpose() * H;

  if (abs(M(0, 1)) + abs(M(0, 2)) + abs(M(1, 0)) + abs(M(1, 2)) + abs(M(2, 0)) +
          abs(M(2, 1)) <
      1e-5) {
    Rest.push_back(H / sqrt(M(0,0)));
    Test.push_back(Eigen::Vector3d(0.0,0.0,0.0));
    return;
  }

  double m1122 = M(0, 0) * M(1, 1);
  double m12sqr = M(0, 1) * M(0, 1);
  double m13sqr = M(0, 2) * M(0, 2);
  double m23sqr = M(1, 2) * M(1, 2);

  double c2 = -(M(0, 0) + M(1, 1) + M(2, 2));
  double c1 = m1122 + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2) -
              (m12sqr + m13sqr + m23sqr);
  double c0 = m12sqr * M(2, 2) + m13sqr * M(1, 1) + m23sqr * M(0, 0) -
              m1122 * M(2, 2) - 2 * M(0, 1) * M(0, 2) * M(1, 2);

  double c22 = c2 * c2;
  double a = c1 - c22 / 3.0;
  double b = (2.0 * c22 * c2 - 9.0 * c2 * c1) / 27.0 + c0;

  double c = 3.0 * b / (2.0 * a) * std::sqrt((-3.0 / a));
  
  if (c < 0.99999) {
    double d = 2.0 * std::sqrt((-a / 3.0));

    double sigma = d * sin(asin(-c) / 3.0) - c2 / 3.0;
    // double sigma =
    //     d * cos(acos(c) / 3.0 - 2.09439510239319526263557236234192) - c2 / 3.0;

    H = H / sqrt(sigma);
    Eigen::Matrix3d Htmp = H;
    bool flag = 0;

    double h1 = H(0, 0);
    double h2 = H(0, 1);
    double h3 = H(0, 2);
    double h4 = H(1, 0);
    double h5 = H(1, 1);
    double h6 = H(1, 2);
    double h7 = H(2, 0);
    double h8 = H(2, 1);
    double h9 = H(2, 2);
    double h11 = h1 * h1;
    double h22 = h2 * h2;
    double h33 = h3 * h3;
    double h44 = h4 * h4;
    double h55 = h5 * h5;
    double h66 = h6 * h6;
    double h77 = h7 * h7;
    double h88 = h8 * h8;
    double h99 = h9 * h9;

    double g1 = h1 * h3 + h4 * h6 + h7 * h9;
    double g2 = h1 * h2 + h4 * h5 + h7 * h8;
    double g3 = h2 * h3 + h5 * h6 + h8 * h9;
    double p0 = h33 + h66 + h99 - 1.0;
    double p1 = -2.0 * g1;
    double p2 = h11 + h44 + h77 - 1.0;

    Eigen::Matrix3d Rrand;
    if (abs(p0) < 1e-8 || abs(p1) < 1e-8 || abs(p2) < 1e-8) { // special planes
      Eigen::Vector4d qgt = Eigen::Quaternion<double>::UnitRandom().coeffs();
      Rrand =
          Eigen::Quaterniond(qgt(0), qgt(1), qgt(2), qgt(3)).toRotationMatrix();
      H = Htmp * (Rrand.transpose());
      h1 = H(0, 0);
      h2 = H(0, 1);
      h3 = H(0, 2);
      h4 = H(1, 0);
      h5 = H(1, 1);
      h6 = H(1, 2);
      h7 = H(2, 0);
      h8 = H(2, 1);
      h9 = H(2, 2);
      h11 = h1 * h1;
      h22 = h2 * h2;
      h33 = h3 * h3;
      h44 = h4 * h4;
      h55 = h5 * h5;
      h66 = h6 * h6;
      h77 = h7 * h7;
      h88 = h8 * h8;
      h99 = h9 * h9;

      g1 = h1 * h3 + h4 * h6 + h7 * h9;
      g2 = h1 * h2 + h4 * h5 + h7 * h8;
      g3 = h2 * h3 + h5 * h6 + h8 * h9;
      p0 = h33 + h66 + h99 - 1.0;
      p1 = -2.0 * g1;
      p2 = h11 + h44 + h77 - 1.0;
      flag = 1;
    }

    double pa[2];
    double qa[2];
    double q0[2];
    double q1[2];
    solve_quadratic(p2, p1, p0, pa);
    for (int i = 0; i < 2; ++i) {
      q0[i] = p0 + pa[i] * p1 / 2.0;
      q1[i] = g2 * pa[i] - g3;
    }
    if (abs(q1[0]) < 1e-8 || abs(q1[1]) < 1e-8 && flag == 0) { // special planes
      Eigen::Vector4d qgt = Eigen::Quaternion<double>::UnitRandom().coeffs();
      Rrand =
          Eigen::Quaterniond(qgt(0), qgt(1), qgt(2), qgt(3)).toRotationMatrix();
      H = Htmp * (Rrand.transpose());
      h1 = H(0, 0);
      h2 = H(0, 1);
      h3 = H(0, 2);
      h4 = H(1, 0);
      h5 = H(1, 1);
      h6 = H(1, 2);
      h7 = H(2, 0);
      h8 = H(2, 1);
      h9 = H(2, 2);
      h11 = h1 * h1;
      h22 = h2 * h2;
      h33 = h3 * h3;
      h44 = h4 * h4;
      h55 = h5 * h5;
      h66 = h6 * h6;
      h77 = h7 * h7;
      h88 = h8 * h8;
      h99 = h9 * h9;

      g1 = h1 * h3 + h4 * h6 + h7 * h9;
      g2 = h1 * h2 + h4 * h5 + h7 * h8;
      g3 = h2 * h3 + h5 * h6 + h8 * h9;
      p0 = h33 + h66 + h99 - 1.0;
      p1 = -2.0 * g1;
      p2 = h11 + h44 + h77 - 1.0;
      flag = 1;
      solve_quadratic(p2, p1, p0, pa);
      for (int i = 0; i < 2; ++i) {
        q0[i] = p0 + pa[i] * p1 / 2.0;
        q1[i] = g2 * pa[i] - g3;
      }
    }

    qa[0] = -q0[0] / q1[0];
    qa[1] = -q0[1] / q1[1];

    double s = 1.0;
    for (int k = 0; k < 2; ++k) {
      double p = pa[k];
      double q = qa[k];

      refine_lambda_s(s, p, q, h1, h2, h3, h4, h5, h6, h7, h8, h9, h11, h22,
                      h33, h44, h55, h66, h77, h88, h99, g1, g2, g3);
      // refine_lambda3(p, q, h1, h2, h3, h4, h5, h6, h7, h8, h9);
      // refine_lambda(p, q, h1, h2, h3, h4, h5, h6, h7, h8, h9);
      Eigen::Vector3d z1(h3 - p * h1, h6 - p * h4, h9 - p * h7);
      Eigen::Vector3d z2(h3 - q * h2, h6 - q * h5, h9 - q * h8);
      Eigen::Matrix3d B;
      B << z1, z2, z1.cross(z2);
      Eigen::Matrix3d A;

      A << -p, 0, q, 0, -q, p, 1.0, 1.0, p * q;
      Eigen::Matrix3d R = B * A.inverse();
      Eigen::Vector3d t(h3 - R(0, 2), h6 - R(1, 2), h9 - R(2, 2));

      Eigen::Vector3d n = A.col(2) / A.col(2).norm();
      t = t / n(2);
      if (flag == 1) {
        R = R * Rrand;
        n = Rrand.transpose() * n;
      }

      if (n(2) < 0)
        n = -n;

      Rest.push_back(R);
      Test.push_back(t);
      Nest.push_back(n);
    }

  } else {
    hdecom_svd(HH, Rest, Test, Nest);
  }
}