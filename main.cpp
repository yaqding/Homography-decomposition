#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core.hpp"

#include "hdec/hsvd.h"
#include "hdec/ding.h"
#include "hdec/malis.h"

const double PI = 3.141592653589793238463;

double rot_error(Eigen::Matrix3d R, Eigen::Matrix3d Rg) {
  double Rnorm =
      std::min((R - Rg).norm() / (2.0 * std::sqrt(2.0)), 0.9999999999999999);
  double error = 2 * asin(Rnorm) * 180.0 / PI;
  return error;
}

double t_error(Eigen::Vector3d T, Eigen::Vector3d Tg) {
  double tnorm = std::min((T / T.norm() - Tg / Tg.norm()).norm(),
                          (T / T.norm() + Tg / Tg.norm()).norm()) /
                 2.0;
  double error =
      2 *
      asin(std::max(std::min(tnorm, 0.9999999999999999), -0.9999999999999999)) *
      180 / PI;
  return error;
}

int main() {

  int num = 1e6;
  int count_ding = 0;
  int count_malis = 0;
  int count_svd = 0;
  int count_cv = 0;
  Eigen::MatrixXd error_ding = Eigen::MatrixXd::Random(num, 2);
  Eigen::MatrixXd error_malis = Eigen::MatrixXd::Random(num, 2);
  Eigen::MatrixXd error_svd = Eigen::MatrixXd::Random(num, 2);
  Eigen::MatrixXd error_cv = Eigen::MatrixXd::Random(num, 2);
  std::vector<long> runtimes_ding;
  std::vector<long> runtimes_malis;
  std::vector<long> runtimes_svd;
  std::vector<long> runtimes_cv;
  for (int i = 0; i < num; ++i) {

    Eigen::Vector4d qgt = Eigen::Quaternion<double>::UnitRandom().coeffs();
    Eigen::Vector3d Tg;
    Tg.setRandom();

    Eigen::Matrix3d Rgt =
        Eigen::Quaterniond(qgt(0), qgt(1), qgt(2), qgt(3)).toRotationMatrix();

    Eigen::Matrix3d X0;
    X0.setRandom();
    Eigen::Matrix3d X1 = Rgt * X0;
    X1.colwise() += Tg;
    Eigen::Matrix3d H = X1 * X0.inverse();

    while (H.determinant() < 0.0) {
      qgt = Eigen::Quaternion<double>::UnitRandom().coeffs();
      Tg.setRandom();
      Rgt =
          Eigen::Quaterniond(qgt(0), qgt(1), qgt(2), qgt(3)).toRotationMatrix();

      X1 = Rgt * X0;
      X1.colwise() += Tg;
      H = X1 * X0.inverse();
    }

    Eigen::Matrix3d HH = H / H.norm();

    std::vector<Eigen::Matrix3d> Rest_malis;
    std::vector<Eigen::Vector3d> Test_malis;
    std::vector<Eigen::Vector3d> Nest_malis;

    auto start_time = std::chrono::high_resolution_clock::now();
    hdecom_malis(HH, Rest_malis, Test_malis, Nest_malis);

    auto end_time = std::chrono::high_resolution_clock::now();
    runtimes_malis.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                             start_time)
            .count());

    std::vector<Eigen::Matrix3d> Rest_ding;
    std::vector<Eigen::Vector3d> Test_ding;
    std::vector<Eigen::Vector3d> Nest_ding;

    start_time = std::chrono::high_resolution_clock::now();
    hdecom_ding(HH, Rest_ding, Test_ding, Nest_ding);

    end_time = std::chrono::high_resolution_clock::now();
    runtimes_ding.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                             start_time)
            .count());

    std::vector<Eigen::Matrix3d> Rest_svd;
    std::vector<Eigen::Vector3d> Test_svd;
    std::vector<Eigen::Vector3d> Nest_svd;
    start_time = std::chrono::high_resolution_clock::now();
    hdecom_svd(HH, Rest_svd, Test_svd, Nest_svd);
    end_time = std::chrono::high_resolution_clock::now();
    runtimes_svd.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               end_time - start_time)
                               .count());

    std::vector<Eigen::Matrix3d> Rest_cv;
    std::vector<Eigen::Vector3d> Test_cv;
    std::vector<Eigen::Vector3d> Nest_cv;

    cv::Mat Hcv;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;
    cv::eigen2cv(HH, Hcv);

    start_time = std::chrono::high_resolution_clock::now();
    int solutions = cv::decomposeHomographyMat(Hcv, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp);
    end_time = std::chrono::high_resolution_clock::now();
    runtimes_cv.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              end_time - start_time)
                              .count());

    for (int k = 0; k < solutions; k=k+2)
      {
        Eigen::Matrix3d Re;
        Eigen::Vector3d Te;
        Eigen::Vector3d Ne;
        cv::cv2eigen(Rs_decomp[k], Re);
        cv::cv2eigen(ts_decomp[k], Te);
        cv::cv2eigen(normals_decomp[k], Ne);
        Rest_cv.push_back(Re);
        Test_cv.push_back(Te);
        Nest_cv.push_back(Ne);
      }


    double tmp_r_ding[2];
    double tmp_r_malis[2];
    double tmp_r_svd[2];
    double tmp_r_cv[2];
    double tmp_t_ding[2];
    double tmp_t_malis[4];
    double tmp_t_svd[2];
    double tmp_t_cv[2];
    for (int k = 0; k < 2; ++k) {
      tmp_r_ding[k] = rot_error(Rest_ding[k], Rgt);
      tmp_r_malis[k] = rot_error(Rest_malis[k], Rgt);
      tmp_r_svd[k] = rot_error(Rest_svd[k], Rgt);
      tmp_r_cv[k] = rot_error(Rest_cv[k], Rgt);
      tmp_t_ding[k] = t_error(Test_ding[k], Tg);
      tmp_t_malis[k] = t_error(Test_malis[k], Tg);
      tmp_t_svd[k] = t_error(Test_svd[k], Tg);
      tmp_t_cv[k] = t_error(Test_cv[k], Tg);
    }

    error_ding(i, 0) = std::min(tmp_r_ding[0], tmp_r_ding[1]);
    error_malis(i, 0) = std::min(tmp_r_malis[0], tmp_r_malis[1]);
    error_svd(i, 0) = std::min(tmp_r_svd[0], tmp_r_svd[1]);
    error_cv(i, 0) = std::min(tmp_r_cv[0], tmp_r_cv[1]);

    error_ding(i, 1) = std::min(tmp_t_ding[0], tmp_t_ding[1]);
    error_malis(i, 1) = std::min(tmp_t_malis[0], tmp_t_malis[1]);
    error_svd(i, 1) = std::min(tmp_t_svd[0], tmp_t_svd[1]);
    error_cv(i, 1) = std::min(tmp_t_cv[0], tmp_t_cv[1]);

    // failure case: rotation or translation error > 0.1 degree
    if (error_ding(i, 0) > 0.01 || error_ding(i, 1) > 0.01) {
      count_ding = count_ding + 1;
    }

    if (error_malis(i, 0) > 0.01 || error_malis(i, 1) > 0.01) {
      count_malis = count_malis + 1;
    }

    if (error_svd(i, 0) > 0.01 || error_svd(i, 1) > 0.01) {
      count_svd = count_svd + 1;
    }

    if (error_cv(i, 0) > 0.01 || error_cv(i, 1) > 0.01) {
      count_cv = count_cv + 1;
    }
  }
  // failure case
  std::cout << "count_ding: " << count_ding << "\n"
            << "count_malis: " << count_malis << "\n"
            << "count_svd: " << count_svd << "\n"
            << "count_cv: " << count_cv << std::endl;
  // running time
  std::sort(runtimes_ding.begin(), runtimes_ding.end());
  std::sort(runtimes_malis.begin(), runtimes_malis.end());
  std::sort(runtimes_svd.begin(), runtimes_svd.end());
  std::sort(runtimes_cv.begin(), runtimes_cv.end());
  std::cout << "Median Run time ding: "
            << runtimes_ding[runtimes_ding.size() / 2] << "ns \n"
            << "Median Run time malis: "
            << runtimes_malis[runtimes_malis.size() / 2] << "ns \n"
            << "Median Run time svd: " << runtimes_svd[runtimes_svd.size() / 2]
            << "ns \n"
            << "Median Run time opencv: " << runtimes_cv[runtimes_cv.size() / 2]
            << "ns" << std::endl;
  std::cout << "Mean Run time ding: "
            << std::accumulate(runtimes_ding.begin(), runtimes_ding.end(),
                               0.0) /
                   runtimes_ding.size()
            << "ns \n"
            << "Mean Run time malis: "
            << std::accumulate(runtimes_malis.begin(), runtimes_malis.end(),
                               0.0) /
                   runtimes_malis.size()
            << "ns \n"
            << "Mean Run time svd: "
            << std::accumulate(runtimes_svd.begin(), runtimes_svd.end(), 0.0) /
                   runtimes_svd.size()
            << "ns \n"
            << "Mean Run time opencv: "
            << std::accumulate(runtimes_cv.begin(), runtimes_cv.end(), 0.0) /
                   runtimes_cv.size()
            << "ns" << std::endl;
  std::cout << "Max Run time ding: " << runtimes_ding[runtimes_ding.size() - 1]
            << "ns \n"
            << "Max Run time malis: "
            << runtimes_malis[runtimes_malis.size() - 1] << "ns \n"
            << "Max Run time svd: " << runtimes_svd[runtimes_svd.size() - 1]
            << "ns \n"
            << "Max Run time opencv: " << runtimes_cv[runtimes_cv.size() - 1]
            << "ns" << std::endl;
  // median rotation error
  Eigen::ArrayXd error_ding_r = error_ding.col(0);
  Eigen::ArrayXd error_malis_r = error_malis.col(0);
  Eigen::ArrayXd error_svd_r = error_svd.col(0);
  Eigen::ArrayXd error_cv_r = error_cv.col(0);
  std::sort(error_ding_r.begin(), error_ding_r.end());
  std::sort(error_malis_r.begin(), error_malis_r.end());
  std::sort(error_svd_r.begin(), error_svd_r.end());
  std::sort(error_cv_r.begin(), error_cv_r.end());
  std::cout << "Median rotation error ding: "
            << error_ding_r[error_ding_r.size() / 2] << "\n"
            << "Median rotation error malis: "
            << error_malis_r[error_malis_r.size() / 2] << "\n"
            << "Median rotation error svd: "
            << error_svd_r[error_svd_r.size() / 2] << "\n"
            << "Median rotation error opencv: "
            << error_cv_r[error_cv_r.size() / 2] << std::endl;
  // mean error
  std::cout << "Mean rotation error ding: " << error_ding_r.mean() << "\n"
            << "Mean rotation error malis: " << error_malis_r.mean() << "\n"
            << "Mean rotation error svd: " << error_svd_r.mean() << "\n"
            << "Mean rotation error opencv: " << error_cv_r.mean() << std::endl;
  // max error
  std::cout << "Max rotation error ding: " << error_ding_r.maxCoeff() << "\n"
            << "Max rotation error malis: " << error_malis_r.maxCoeff() << "\n"
            << "Max rotation error svd: " << error_svd_r.maxCoeff() << "\n"
            << "Max rotation error opencv: " << error_cv_r.maxCoeff()
            << std::endl;

  // translation error
  Eigen::ArrayXd error_ding_t = error_ding.col(1);
  Eigen::ArrayXd error_malis_t = error_malis.col(1);
  Eigen::ArrayXd error_svd_t = error_svd.col(1);
  Eigen::ArrayXd error_cv_t = error_cv.col(1);
  std::sort(error_ding_t.begin(), error_ding_t.end());
  std::sort(error_malis_t.begin(), error_malis_t.end());
  std::sort(error_svd_t.begin(), error_svd_t.end());
  std::sort(error_cv_t.begin(), error_cv_t.end());
  std::cout << "Median translation error ding: "
            << error_ding_t[error_ding_t.size() / 2] << "\n"
            << "Median translation error malis: "
            << error_malis_t[error_malis_t.size() / 2] << "\n"
            << "Median translation error svd: "
            << error_svd_t[error_svd_t.size() / 2] << "\n"
            << "Median translation error opencv: "
            << error_cv_t[error_cv_t.size() / 2] << std::endl;

  std::cout << "Mean translation error ding: " << error_ding_t.mean() << "\n"
            << "Mean translation error malis: " << error_malis_t.mean() << "\n"
            << "Mean translation error svd: " << error_svd_t.mean() << "\n"
            << "Mean translation error opencv: " << error_cv_t.mean()
            << std::endl;

  std::cout << "Max translation error ding: " << error_ding_t.maxCoeff() << "\n"
            << "Max translation error malis: " << error_malis_t.maxCoeff()
            << "\n"
            << "Max translation error svd: " << error_svd_t.maxCoeff() << "\n"
            << "Max translation error opencv: " << error_cv_t.maxCoeff()
            << std::endl;
}
