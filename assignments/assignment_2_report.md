# Instructions for running code
To run the main code 
uv run assignment_2.py 

# 1. Sketches of inverted pendulum

Included at the end of document

# 2. Visualization of the regions of attraction

At the end of report

# 3. Choice of Poincare section

The poincare section I am using is when theta is 0. This is a good poincare section because theta 0  is parallel to the paths and the paths are generally similarly spaced. Theta 0 is also an easy place to analyize and far from the impacts so they won't have a significant efffect. Using a poincare section also reduces the problem to a single state because the only free variable is angular velocity. 

# 4. Grid Resolution Varification

I created a lookup table that uses the nearest grid state to choose each step. The grid has to be fine enough that the rounding consistently gives accurate results but runs quickly. I picked 50 random starting points to test each grid resolution and simulated to see how many steps they took to reach a standstill and if they took the expected amount of steps. I used q 80% threshold for both categories because some of the values near the edge of the first node could not reach node 0. 

| n_velocity | n_alpha | velocity spacing (rad/s) | alpha spacing (rad) | reaches RoA | steps as predicted | passes | steps from 3.0(predicted / actual) |
|---|---|---|---|---|---|---|---|
| 120 | 60 | 0.037 | 0.00095 | 65.2% | 65.2% | no | 3 / inf |
| 180 | 60 | 0.025 | 0.00095 | 67.4% | 67.4% | no | 3 / 3 |
| 240 | 60 | 0.019 | 0.00095 | 84.8% | 84.8% | yes | 3 / 3 |
| 360 | 60 | 0.012 | 0.00095 | 84.8% | 84.8% | yes | 3 / 3 |
| 480 | 60 | 0.009 | 0.00095 | 91.3% | 91.3% | yes | 3 / 3 |

| n_velocity | n_alpha | velocity spacing (rad/s) | alpha spacing (rad) | reaches RoA | steps as predicted | passes | steps from 3.0(predicted / actual) |
|---|---|---|---|---|---|---|---|
| 240 | 20 | 0.019 | 0.00295 | 80.4% | 76.1% | no | 3 / 3 |
| 240 | 40 | 0.019 | 0.00144 | 71.7% | 71.7% | no | 3 / 3 |
| 240 | 50 | 0.019 | 0.00114 | 84.8% | 82.6% | yes | 3 / 3 |
| 240 | 60 | 0.019 | 0.00095 | 84.8% | 84.8% | yes | 3 / 3 |


I decided to use n_velocity = 240 and n_alpha = 50 for my grid. This grid value reached my desired threshold and every finer grid i tested also passed.

# 5. Plot the trajectory for max and min steps

At the end of report

# 6. Visulaization of how many steps

At the end of report

