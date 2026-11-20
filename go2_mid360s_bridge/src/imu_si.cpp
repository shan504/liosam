#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

using Imu = sensor_msgs::msg::Imu;

class LivoxImuSI : public rclcpp::Node
{
public:
  LivoxImuSI() : Node("livox_imu_si")
  {
    pub_ = create_publisher<Imu>("/mid360s/imu_raw_si", rclcpp::SensorDataQoS());
    sub_ = create_subscription<Imu>("/livox/imu", rclcpp::SensorDataQoS(),
      [this](Imu::ConstSharedPtr input) {
        Imu output = *input;
        constexpr double gravity = 9.80665;
        auto & a = output.linear_acceleration;
        a.x *= gravity;
        a.y *= gravity;
        a.z *= gravity;
        if (output.linear_acceleration_covariance[0] >= 0.0) {
          for (auto & value : output.linear_acceleration_covariance) {
            value *= gravity * gravity;
          }
        }
        // Driver data is raw gyro + accelerometer. Madgwick will estimate attitude.
        output.orientation.x = output.orientation.y = output.orientation.z = 0.0;
        output.orientation.w = 1.0;
        output.orientation_covariance.fill(0.0);
        output.orientation_covariance[0] = -1.0;
        pub_->publish(output);
      });
  }

private:
  rclcpp::Publisher<Imu>::SharedPtr pub_;
  rclcpp::Subscription<Imu>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LivoxImuSI>());
  rclcpp::shutdown();
  return 0;
}
