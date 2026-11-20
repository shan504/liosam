#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "cloud_points.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;
using Input = livox_ros_driver2::msg::CustomMsg;
using go2_lio::Point;

Field field(const char * name, std::uint32_t offset, std::uint8_t type)
{
  Field f;
  f.name = name;
  f.offset = offset;
  f.datatype = type;
  f.count = 1;
  return f;
}

class CloudAdapter : public rclcpp::Node
{
public:
  CloudAdapter() : Node("mid360s_cloud_adapter")
  {
    output_ = create_publisher<Cloud>("/mid360s/points", rclcpp::SensorDataQoS());
    input_ = create_subscription<Input>("/livox/lidar", rclcpp::SensorDataQoS(),
      [this](Input::ConstSharedPtr msg) { publish_cloud(*msg); });
  }

private:
  void publish_cloud(const Input & msg)
  {
    auto points = go2_lio::convert_points(msg.points);
    if (points.empty()) {
      return;
    }
    Cloud cloud;
    cloud.header = msg.header;
    cloud.height = 1;
    cloud.width = static_cast<std::uint32_t>(points.size());
    cloud.fields = {
      field("x", offsetof(Point, x), Field::FLOAT32),
      field("y", offsetof(Point, y), Field::FLOAT32),
      field("z", offsetof(Point, z), Field::FLOAT32),
      field("intensity", offsetof(Point, intensity), Field::FLOAT32),
      field("time", offsetof(Point, time), Field::FLOAT32),
      field("ring", offsetof(Point, ring), Field::UINT16)
    };
    const std::uint16_t one = 1;
    cloud.is_bigendian = *reinterpret_cast<const unsigned char *>(&one) == 0;
    cloud.is_dense = true;
    cloud.point_step = sizeof(Point);
    cloud.row_step = cloud.width * cloud.point_step;
    cloud.data.resize(cloud.row_step);
    std::memcpy(cloud.data.data(), points.data(), cloud.data.size());
    output_->publish(cloud);
  }

  rclcpp::Publisher<Cloud>::SharedPtr output_;
  rclcpp::Subscription<Input>::SharedPtr input_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudAdapter>());
  rclcpp::shutdown();
  return 0;
}
