#! /usr/bin/env python
# -*- encoding: UTF-8 -*-


'''
 *GEM2 - Gait-phase Estimation Module 2
 *
 * Copyright 2020-2021 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *	 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
'''



import rospy
from gem2 import GEM2
from gem2_tools import GEM2_tools
from gem2_tools import GEM2_data
import numpy as np
from std_msgs.msg import Int32 
from std_msgs.msg import Float32 
from std_msgs.msg import String 
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu
import pickle as pickle


class  gem2_ros():
	def __init__(self):
		rospy.init_node("gait_phase_estimation_module")
		self.phase_pub = rospy.Publisher('/gem2/support_phase', Int32, queue_size=1000)
		self.leg_pub = rospy.Publisher('/gem2/support_leg', String, queue_size=1000)
		self.RLeg_pub = rospy.Publisher('/gem2/RLegContactProbability', Float32, queue_size=1000)
		self.LLeg_pub = rospy.Publisher('/gem2/LLegContactProbability', Float32, queue_size=1000)
		freq = rospy.get_param('gem2_freq',200)
		self.freq = freq
		self.phase_msg = Int32()
		self.phase = -1	
		self.support_msg = String()
		self.LLeg_msg = Float32()
		self.RLeg_msg = Float32()
		self.support_leg = "None"
		robot = rospy.get_param('gem2_robot','nao')
		print('Loading the GEM2 Models')
		imu_topic = rospy.get_param('gem2_base_imu_topic')
		limu_topic = rospy.get_param('gem2_left_leg_imu_topic')
		rimu_topic = rospy.get_param('gem2_right_leg_imu_topic')

		vcom_topic = rospy.get_param('gem2_com_velocity_topic')
		lft_topic = rospy.get_param('gem2_left_leg_wrench_topic')
		rft_topic = rospy.get_param('gem2_right_leg_wrench_topic')
		lvel_topic = rospy.get_param('gem2_left_leg_velocity_topic')
		rvel_topic = rospy.get_param('gem2_right_leg_velocity_topic')


		self.vcom_sub  = rospy.Subscriber(vcom_topic,TwistStamped, self.vcomcb)
		self.imu_sub   = rospy.Subscriber(imu_topic,Imu,  self.imucb)
		self.lft_sub  = rospy.Subscriber(lft_topic, WrenchStamped, self.lwrenchcb)
		self.rft_sub  = rospy.Subscriber(rft_topic, WrenchStamped, self.rwrenchcb)
		self.lvel_sub  = rospy.Subscriber(lvel_topic,TwistStamped, self.lvelcb)
		self.rvel_sub  = rospy.Subscriber(rvel_topic,TwistStamped, self.rvelcb)
		self.limu_sub   = rospy.Subscriber(limu_topic,Imu,  self.limucb)
		self.rimu_sub   = rospy.Subscriber(rimu_topic,Imu,  self.rimucb)

		
		
		
		self.lwrench_inc = False
		self.rwrench_inc = False
		self.imu_inc = False
		self.limu_inc = False
		self.rimu_inc = False
		self.vcom_inc = False
		self.lvel_inc = False
		self.rvel_inc = False

		self.g = pickle.load(open(robot + '_gem2.sav','rb'))
		self.gt = pickle.load(open(robot + '_gem2_tools.sav', 'rb'))
	

		print('Gait-Phase Estimation Module Initialized Successfully')
	
	def lwrenchcb(self,msg):
		self.lwrench = msg
		self.lwrench_inc = True

	def rwrenchcb(self,msg):
		self.rwrench = msg
		self.rwrench_inc = True

	def vcomcb(self,msg):
		self.vcom = msg
		self.vcom_inc = True

	def lvelcb(self,msg):
		self.lvel = msg
		self.lvel_inc = True

	def rvelcb(self,msg):
		self.rvel = msg
		self.rvel_inc = True

	def imucb(self,msg):
		self.imu = msg
		self.imu_inc = True

	def rimucb(self,msg):
		self.rimu = msg
		self.rimu_inc = True

	def limucb(self,msg):
		self.limu = msg
		self.limu_inc = True

	def predictUL(self):
		if(self.imu_inc and  self.limu_inc and self.rimu_inc and self.lwrench_inc and self.rwrench_inc and self.vcom_inc and self.rvel_inc and self.lvel_inc):
			self.imu_inc = False
			self.limu_inc = False
			self.rimu_inc = False
			self.lwrench_inc = False
			self.rwrench_inc = False
			self.vcom_inc = False
			self.lvel_inc = False
			self.rvel_inc = False
			data = GEM2_data()
			#Leg Ground Reaction Forces (F/T)
			data.lfX =  self.lwrench.wrench.force.x
			data.lfY =  self.lwrench.wrench.force.y 
			data.lfZ =  self.lwrench.wrench.force.z 
			data.rfX =  self.rwrench.wrench.force.x
			data.rfY =  self.rwrench.wrench.force.y 
			data.rfZ =  self.rwrench.wrench.force.z 
			#Leg Ground Reaction Torques (F/T)
			data.ltX =  self.lwrench.wrench.torque.x
			data.ltY =  self.lwrench.wrench.torque.y 
			data.ltZ =  self.lwrench.wrench.torque.z 
			data.rtX =  self.rwrench.wrench.torque.x
			data.rtY =  self.rwrench.wrench.torque.y 
			data.rtZ =  self.rwrench.wrench.torque.z 
			#Base IMU
			data.accX = self.imu.linear_acceleration.x
			data.accY = self.imu.linear_acceleration.y
			data.accZ = self.imu.linear_acceleration.z
			data.gX = self.imu.angular_velocity.x
			data.gY = self.imu.angular_velocity.y
			data.gZ = self.imu.angular_velocity.z
			#Left Leg IMU
			data.laccX = self.limu.linear_acceleration.x
			data.laccY = self.limu.linear_acceleration.y
			data.laccZ = self.limu.linear_acceleration.z
			data.lgX = self.limu.angular_velocity.x
			data.lgY = self.limu.angular_velocity.y
			data.lgZ = self.limu.angular_velocity.z
			#Right Leg IMU
			data.raccX = self.rimu.linear_acceleration.x
			data.raccY = self.rimu.linear_acceleration.y
			data.raccZ = self.rimu.linear_acceleration.z
			data.rgX = self.rimu.angular_velocity.x
			data.rgY = self.rimu.angular_velocity.y
			data.rgZ = self.rimu.angular_velocity.z
			#CoM velocity (Kinematics)
			data.dcX = self.vcom.twist.linear.x
			data.dcY = self.vcom.twist.linear.y
			data.dcZ = self.vcom.twist.linear.z
			#Left Leg Relative Velocity (Kinematics)
			data.lvX = self.lvel.twist.linear.x
			data.lvY = self.lvel.twist.linear.y
			data.lvZ = self.lvel.twist.linear.z
			data.lwX = self.lvel.twist.angular.x
			data.lwY = self.lvel.twist.angular.y
			data.lwZ = self.lvel.twist.angular.z
			#Right Leg Relative Velocity (Kinematics)
			data.rvX = self.rvel.twist.linear.x
			data.rvY = self.rvel.twist.linear.y
			data.rvZ = self.rvel.twist.linear.z
			data.rwX = self.rvel.twist.angular.x
			data.rwY = self.rvel.twist.angular.y
			data.rwZ = self.rvel.twist.angular.z

			self.phase, self.reduced_data = self.g.predict(self.gt.genInpu(data,self.gt))

			self.support_leg = self.g.getSupportLeg()
			self.phase_msg.data = self.phase        	
			self.phase_pub.publish(self.phase_msg)
			self.support_msg.data = self.support_leg        	
			self.leg_pub.publish(self.support_msg)

	

	def run(self):
		r = rospy.Rate(self.freq) 
		while not rospy.is_shutdown():
			self.predictUL()
			r.sleep()
   		
if __name__ == "__main__":
	g = gem2_ros()
	try:
		g.run()
	except rospy.ROSInterruptException:
		pass
