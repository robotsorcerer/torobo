# from execute cart to joint


	def cart_to_joint(self, cart_pos, timeout=0.5, check=True):

		ik_solver = TRAC_IK("link0", "link7", self.urdf,
							ik_config['ik_timeout'],  # default seconds
							1e-5,   # default epsilon
							"Speed")

		qinit = [-1.24, -8.16, -4.30, 68.21, -17.01, 59.76, 0.03]#
		x, y, z = cart_pos

		rx = ry = rz = 0.0
		rw = 1.0
		bx = by = bz = 0.001
		brx = bry = brz = 0.1

		avg_time = num_solutions_found = 0

		for i in range(ik_config['ik_trials_num']):
			ini_t = time.time()
			pos = ik_solver.CartToJnt(qinit,
									  x, y, z,
									  rx, ry, rz, rw,
									  bx, by, bz,
									  brx, bry, brz)
			fin_t = time.time()
			call_time = fin_t - ini_t
			avg_time += call_time

			if pos:# and vel:
				num_solutions_found += 1

		if check:
			print("Average IK call time: %.4f mins"%(avg_time*60))

		return list(pos)

	"""
	def cart_to_joint(self, cart_pos, timeout=ik_config['ik_timeout'], check=True):

		ik_solver = IK(base_link="link0", tip_link="link7",
							timeout=timeout,  # default seconds
							epsilon=1e-5,   # default epsilon
							solve_type="Speed",
							urdf_string=self.urdf)

		seed_state = [0.0] * ik_solver.number_of_joints
		seed_state = [-1.24, -8.16, -4.30, 68.21, -17.01, 59.76, 0.03]

		x, y, z = cart_pos

		joint_angles = ik_solver.get_ik(seed_state, \
										x, y, z, \
						                0.0, 0.0, 0.0, 1.0, \
										0.01, 0.01, 0.01,   # X, Y, Z bounds
										0.1, 0.1, 0.1)   # Rotation X, Y, Z bounds

		print('joint_angles: ', joint_angles)
		return list(joint_angles)
	"""
