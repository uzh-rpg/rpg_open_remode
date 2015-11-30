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

#include <rmd/depthmap.h>

#include <rmd/publisher.h>

#include <svo_msgs/DenseInput.h>
#include <ros/ros.h>

namespace rmd
{

namespace ProcessingStates
{
enum State
{
  UPDATE,
  TAKE_REFERENCE_FRAME,
};
}
typedef ProcessingStates::State State;

class DepthmapNode
{
public:
  DepthmapNode(ros::NodeHandle &nh);
  bool init();
  void denseInputCallback(
      const svo_msgs::DenseInputConstPtr &dense_input);
private:
  void denoiseAndPublishResults();
  void publishConvergenceMap();

  std::shared_ptr<rmd::Depthmap> depthmap_;
  State state_;
  float ref_compl_perc_;
  float max_dist_from_ref_;
  int publish_conv_every_n_;
  int num_msgs_;

  ros::NodeHandle &nh_;
  std::unique_ptr<rmd::Publisher> publisher_;
};

} // rmd namespace

#endif // RMD_DEPTHMAP_NODE_H
