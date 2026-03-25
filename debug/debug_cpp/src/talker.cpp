#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class Talker : public rclcpp::Node {
public:
  Talker() : Node("debug_cpp_talker"), count_(0) {
    publisher_ = this->create_publisher<std_msgs::msg::String>("debug_topic", 10);
    timer_ = this->create_wall_timer(1s, [this]() { timer_callback(); });
    RCLCPP_INFO(this->get_logger(), "C++ Talker node started");
  }

private:
  void timer_callback() {
    auto msg = std_msgs::msg::String();
    msg.data = "Hello from debug_cpp [" + std::to_string(count_++) + "]";
    publisher_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", msg.data.c_str());
  }

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  size_t count_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Talker>());
  rclcpp::shutdown();
  return 0;
}
