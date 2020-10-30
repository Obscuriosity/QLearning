QL_256_Explore

A new Q Learning program which will use all the sonar individually so will have eight states.
The states will be either 0 or 1 registering close range contact nearer than 15cm.
Eight states, either 0 or 1 gives 256 states in total.
The three actions will be spin on the spot left, spin right or drive forward.
256 x 3 = 768 entries in the q table.
My experience has been of much failure in long state lists but I will give it a go.

The idea is that when the obstacles are far off, the agent, Terence, will refer to the QL_16_Explore Q table which should keep it from crashing.
When the world gets closer due to QL_16_Explore failing, Terence will refer to a higher definition picture of the environment and use this new Q Table.

That's the idea anyway.
