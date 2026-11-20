#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace go2_lio
{
struct Point
{
  float x, y, z, intensity, time;
  std::uint16_t ring, padding;
};

static_assert(sizeof(Point) == 24 && std::is_trivially_copyable<Point>::value,
              "PointCloud2 binary layout must be 24 bytes per point");

template<typename Range>
std::vector<Point> convert_points(const Range & source)
{
  std::vector<Point> result;
  result.reserve(source.size());
  for (const auto & p : source) {
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
      result.push_back({p.x, p.y, p.z, static_cast<float>(p.reflectivity),
                        static_cast<float>(p.offset_time * 1e-9), p.line, 0});
    }
  }
  // Upstream uses the final point's time as the scan-end offset for deskew.
  const auto earlier = [](const Point & a, const Point & b) { return a.time < b.time; };
  if (!std::is_sorted(result.begin(), result.end(), earlier)) {
    std::stable_sort(result.begin(), result.end(), earlier);
  }
  return result;
}
}  // namespace go2_lio
