| run_id | layers | levers | trainable_pct | peak_gb | mem_status | mem_reduction_pct | heldout_loss | d_task_pts | tok_s | step_time_s | pareto | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top8_q4_gc | `................########` | NF4+GC+chunk |  | 0.9509 | measured | 89.9 | 1.688 | -2.61 | 3841 |  | **yes** | done |
| lora_all_q4_gc | `########################` | NF4+GC+chunk |  | 1.071 | measured | 88.6 |  |  | 1868 |  |  | incomplete |
| top4 | `....................####` | chunk |  | 2.205 | measured | 76.6 |  |  | 7104 |  |  | incomplete |
| top8 | `................########` | chunk |  | 3.039 | measured | 67.7 | 1.612 | -0.45 | 5822 |  | **yes** | done |
| budget35 | `...............#########` | chunk |  | 3.247 | measured | 65.5 | 1.605 | -0.32 | 5806 |  | **yes** | done |
| top12 | `............############` | chunk |  | 3.872 | measured | 58.9 |  |  | 4896 |  |  | incomplete |
| budget50 | `...........#######.###.#` | chunk |  | 3.953 | measured | 58 | 1.589 | -0.67 | 5143 |  | **yes** | done |
| random4 | `#..#................#..#` | chunk |  | 5.096 | measured | 45.8 | 1.591 | 1.36 | 4495 |  |  | done |
| imp4 | `#.#........##...........` | chunk |  | 5.096 | measured | 45.8 | 1.602 | -0.35 | 5003 |  |  | done |
| drift4 | `###...#.................` | chunk |  | 5.096 | measured | 45.8 | 1.606 | 1.57 | 5341 |  |  | done |
| random8 | `#..##..##..........##.#.` | chunk |  | 5.351 | measured | 43.1 | 1.583 | 3.12 | 4171 |  | **yes** | done |
| imp8 | `#.###....####...........` | chunk |  | 5.351 | measured | 43.1 | 1.597 | -0.35 | 4675 |  |  | done |
| drift8 | `####.##..............#.#` | chunk |  | 5.351 | measured | 43.1 | 1.587 | 2.35 | 4908 |  |  | done |
| bottom8 | `########................` | chunk |  | 5.351 | measured | 43.1 | 1.602 | 3.01 | 4140 |  |  | done |
| random12 | `#.###.####.........####.` | chunk |  | 5.607 | measured | 40.4 | 1.58 | 2.24 | 3879 |  | **yes** | done |
| imp12 | `#######.#####...........` | chunk |  | 5.607 | measured | 40.4 | 1.595 | 0.22 | 4248 |  |  | done |
| drift12 | `########...........###.#` | chunk |  | 5.607 | measured | 40.4 | 1.58 | 2.67 | 4459 |  |  | done |
| lora_all | `########################` | chunk |  | 6.372 | measured | 32.3 | 1.571 | 0.77 | 3331 |  | **yes** | done |
| lora_all_plain | `########################` | plain |  | 9.41 | measured | 0 | 1.575 | 0.32 | 2252 |  |  | done |

Baseline for memory reduction: `lora_all_plain` (9.4099 GB). Reference for Δtask: `phase1_ref` (held-out loss 1.7931028370125812). Pareto computed on measured (isolated) memory only; 'inflated' rows need `python scripts/reprobe.py --auto`.
