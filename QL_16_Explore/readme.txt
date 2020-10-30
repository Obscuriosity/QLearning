# QL_16_Explore
Robot Q Learning experiment with 16 states and 3 actions.

States are 0 or 1 on four sonar to imply objects are within 36cm
Actions are forward, turn left- pivoting on left wheel and turn right pivoting on right wheel.
No reward adjustment for forward motion, small negative adjustment for turning left or right.
Will persist with negative reward proportional to distance from obstruction.
will try large negative reward for crash if the above does not work.
