///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2018, The Regents of the University of California
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <flute.hpp>
#include <tuple>

namespace pdr {
using namespace flt;
typedef long long int64_t;
Tree primDijkstra(const std::vector<int>& x,
                       const std::vector<int>& y,
                       int driver_index,
                       float alpha);
class Point
{
 public:
  Point() = default;
  Point(const Point& p) = default;
  Point(int x, int y){
    x_ = x;
    y_ = y;
  }
  ~Point() = default;
  Point& operator=(const Point& rhs) = default;
  bool operator==(const Point& rhs) const{return std::tie(x_, y_) == std::tie(rhs.x_, rhs.y_);}
  bool operator!=(const Point& rhs) const { return !(*this == rhs); };
  bool operator<(const Point& rhs) const {  return std::tie(x_, y_) < std::tie(rhs.x_, rhs.y_);}
  bool operator>=(const Point& rhs) const { return !(*this < rhs); }
  int getX() const { return x_; }
  int getY() const { return y_; }
  void setX(int x) { x_ = x; }
  void setY(int y) { y_ = y; }
  void addX(int x) { x_ += x; }
  void addY(int y) { y_ += y; }


  int x() const { return x_; }
  int y() const { return y_; }


  // compute the manhattan distance between two points
    static int64_t manhattanDistance(Point p0, Point p1)
    {
        const int64_t dx = std::abs(p1.x_ - p0.x_);
        const int64_t dy = std::abs(p1.y_ - p0.y_);
        return dx + dy;
    }

//   friend dbIStream& operator>>(dbIStream& stream, Point& p);
//   friend dbOStream& operator<<(dbOStream& stream, const Point& p);

 private:
  int x_ = 0;
  int y_ = 0;
};
}  // namespace pdr
