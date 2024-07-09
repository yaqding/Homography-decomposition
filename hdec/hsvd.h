#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

void hdecom_svd(Eigen::Matrix3d &HH, std::vector<Eigen::Matrix3d> &Rest, std::vector<Eigen::Vector3d> &Test, std::vector<Eigen::Vector3d> &Nest) {

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(HH, Eigen::ComputeFullV);
  Eigen::Matrix3d H2 = HH / svd.singularValues()[1];

  Eigen::Vector3d S2 = svd.singularValues();
  Eigen::Matrix3d V2 = svd.matrixV();

  if (abs(S2(0) - S2(2)) < 1.0e-6)
  {
    Rest.push_back(H2);
    Test.push_back(Eigen::Vector3d(0.0,0.0,0.0));
    return;
  }

  if (V2.determinant() < 0) {
    V2 *= -1;
  }

  double s1 = S2(0) * S2(0) / (S2(1) * S2(1));
  double s3 = S2(2) * S2(2) / (S2(1) * S2(1));

  Eigen::Vector3d v1 = V2.col(0);
  Eigen::Vector3d v2 = V2.col(1);
  Eigen::Vector3d v3 = V2.col(2);

  Eigen::Vector3d u1 = (std::sqrt(1.0 - s3) * v1 + std::sqrt(s1 - 1.0) * v3) /
                       std::sqrt(s1 - s3);
  Eigen::Vector3d u2 = (std::sqrt(1.0 - s3) * v1 - std::sqrt(s1 - 1.0) * v3) /
                       std::sqrt(s1 - s3);

  Eigen::Matrix3d U1;
  Eigen::Matrix3d W1;
  Eigen::Matrix3d U2;
  Eigen::Matrix3d W2;
  U1.col(0) = v2;
  U1.col(1) = u1;
  U1.col(2) = v2.cross(u1);

  W1.col(0) = H2 * v2;
  W1.col(1) = H2 * u1;
  W1.col(2) = (H2 * v2).cross(H2 * u1);

  U2.col(0) = v2;
  U2.col(1) = u2;
  U2.col(2) = v2.cross(u2);

  W2.col(0) = H2 * v2;
  W2.col(1) = H2 * u2;
  W2.col(2) = (H2 * v2).cross(H2 * u2);

  // # compute the rotation matrices
  Eigen::Matrix3d R1 = W1 * U1.transpose();
  Eigen::Matrix3d R2 = W2 * U2.transpose();

  Eigen::Vector3d n1 = v2.cross(u1);
  
  if (n1(2) < 0)
  {
    n1 = -n1;
  }
  Eigen::Vector3d t1 = (H2 - R1) * n1;

  Eigen::Vector3d n2 = v2.cross(u2);
  
  if (n2(2) < 0)
  {
    n2 = -n2;
  }
  Eigen::Vector3d t2 = (H2 - R2) * n2;

  Rest.push_back(R1);
  Rest.push_back(R2);
  Test.push_back(t1);
  Test.push_back(t2);
  Nest.push_back(n1);
  Nest.push_back(n2);
  
}
