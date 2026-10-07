# Outcome-decomposition findings (audit fix, Task 2)

CTE and collision incidence, split by episode outcome, to remove the
selection/composition artifact identified in the validity audit.
In every study `endOnCollision=false` and there is no off-channel
termination, so the **only failure mode is timeout** (`truncated==1`);
collisions are a within-episode safety metric orthogonal to success.


## Study 2 (robustness)

- other (non-timeout) failures across all cells: 0 (expected 0 — confirms timeout is the sole failure mode).
- `DQN 0.0 0.0`: success 17.2%, CTE all=28.26 / success=0.00 / fail=30.20; collision-rate all=2.8% / on-success=0.0%.
- `DQN 0.25 0.4`: success 36.7%, CTE all=27.72 / success=20.40 / fail=32.88; collision-rate all=43.9% / on-success=43.3%.
- `DoubleDQN 0.0 0.0`: success 25.6%, CTE all=43.25 / success=0.00 / fail=54.88; collision-rate all=10.6% / on-success=6.7%.
- `DoubleDQN 0.25 0.4`: success 22.2%, CTE all=43.64 / success=19.61 / fail=49.79; collision-rate all=30.6% / on-success=35.2%.
- `DuelingDQN 0.0 0.0`: success 41.7%, CTE all=14.97 / success=0.00 / fail=23.64; collision-rate all=18.9% / on-success=18.8%.
- `DuelingDQN 0.25 0.4`: success 22.8%, CTE all=45.52 / success=15.31 / fail=51.10; collision-rate all=37.2% / on-success=26.2%.
- `DuelingDoubleDQN 0.0 0.0`: success 23.3%, CTE all=20.92 / success=0.00 / fail=27.43; collision-rate all=6.7% / on-success=8.3%.
- `DuelingDoubleDQN 0.25 0.4`: success 27.2%, CTE all=35.69 / success=15.49 / fail=41.98; collision-rate all=46.7% / on-success=28.6%.

## Study 3 (rule-based + confirmatory)

- other (non-timeout) failures across all cells: 0 (expected 0 — confirms timeout is the sole failure mode).
- `DQN clean`: success 20.0%, CTE all=28.81 / success=0.38 / fail=31.67; collision-rate all=5.6% / on-success=5.9%.
- `DQN harsh`: success 32.0%, CTE all=31.99 / success=17.84 / fail=37.55; collision-rate all=42.4% / on-success=46.6%.
- `DoubleDQN clean`: success 19.3%, CTE all=29.02 / success=0.71 / fail=34.89; collision-rate all=6.0% / on-success=5.6%.
- `DoubleDQN harsh`: success 23.8%, CTE all=42.25 / success=20.50 / fail=47.10; collision-rate all=37.8% / on-success=41.2%.
- `DuelingDQN clean`: success 32.2%, CTE all=25.58 / success=0.00 / fail=31.43; collision-rate all=10.2% / on-success=14.0%.
- `DuelingDQN harsh`: success 32.7%, CTE all=35.58 / success=15.75 / fail=40.70; collision-rate all=44.2% / on-success=44.7%.
- `DuelingDoubleDQN clean`: success 17.8%, CTE all=27.64 / success=0.12 / fail=31.83; collision-rate all=4.7% / on-success=3.8%.
- `DuelingDoubleDQN harsh`: success 29.6%, CTE all=35.96 / success=15.29 / fail=40.98; collision-rate all=45.8% / on-success=34.9%.
- `RuleBased clean`: success 47.3%, CTE all=11.00 / success=0.00 / fail=23.68; collision-rate all=0.0% / on-success=0.0%.
- `RuleBased harsh`: success 47.3%, CTE all=22.83 / success=13.16 / fail=34.13; collision-rate all=34.2% / on-success=34.5%.
