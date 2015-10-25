// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RMD_DEPTHMAP_NODE_H
#define RMD_DEPTHMAP_NODE_H

#include <ros/ros.h>
#include <svo_msgs/DenseInput.h>

#include <rmd/depthmap.h>

namespace rmd
{

class DepthmapNode
{
public:
  void denseInputCallback(
      const svo_msgs::DenseInputConstPtr &dense_input);
};

} // rmd namespace

#endif // RMD_DEPTHMAP_NODE_H
