| run_id | layers | levers | trainable_pct | peak_gb | mem_status | mem_reduction_pct | heldout_loss | d_task_pts | tok_s | step_time_s | pareto | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spec_C_grad6_gc | `###.#......##...........` | NF4+GC+chunk | 0.0857 | 0.7824 | measured | 32.4 | 1.752 | -0.59 | 451.8 | 2.326 | **yes** | done |
| spec_D_grad6_gc_offload | `###.#......##...........` | NF4+GC+chunk+offload | 0.0857 | 1.024 | measured | 11.4 | 1.752 | -0.59 | 395.6 | 2.657 |  | done |
| spec_E_grad3 | `#.#.#...................` | NF4+chunk | 0.0429 | 1.124 | measured | 2.8 | 1.769 | -0.25 | 748.5 | 1.404 |  | done |
| spec_B_grad6 | `###.#......##...........` | NF4+chunk | 0.0857 | 1.129 | measured | 2.4 | 1.752 | -0.59 | 730.7 | 1.438 |  | done |
| spec_E_grad12 | `#####.#.#####...#.......` | NF4+chunk | 0.1713 | 1.138 | measured | 1.6 | 1.737 | -0.87 | 714.8 | 1.47 | **yes** | done |
| spec_A_all | `########################` | NF4+chunk | 0.342 | 1.157 | measured | 0 | 1.714 | -0.15 | 636.2 | 1.652 | **yes** | done |

Baseline for memory reduction: `spec_A_all` (1.1568 GB). Reference for Δtask: `spec_ref` (held-out loss 1.7951911256359125). Pareto computed on measured (isolated) memory only; 'inflated' rows need `python scripts/reprobe.py --auto`.
