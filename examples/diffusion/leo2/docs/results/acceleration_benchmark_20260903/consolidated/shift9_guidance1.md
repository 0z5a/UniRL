# Leo2 acceleration results: flow shift 9, guidance 1

MSE is mean squared error over all equal-size samples; global RMSE is its square root.
Relative L1/L2 rows are means of per-video norm ratios. `N/A` means the source did not record a value or provenance marks it unavailable.
Speedup intervals are normal-approximation 95% CIs across 16 prompt-paired ratios, not repeated-run uncertainty.
VBench scores use a 16-video custom-input suite, not the leaderboard; dynamic degree is descriptive, not monotonic quality.

| Metric | Unit | exact | static_0.10 | taylor_0.10_m0.50 | magcache_0.12_k4 | dfr_w10_49_i2 |
|---|---|---:|---:|---:|---:|---:|
| Artifact case name | text | cache_off_shift9 | cache_t010_shift9 | taylor_t010_m050_shift9 | magcache_t012_k4_r020_shift9 | dfr_w10_49_i2_shift9 |
| Acceleration method | text | off | first_block | taylor | magcache | fastercache_dfr |
| Acceleration method options | JSON | {} | {"threshold":0.1} | {"max_extrapolation":0.5,"threshold":0.1} | {"calibrate":false,"max_skip_steps":4,"profile":"/root/leo2-output/accel-magcache-calibration-shift9-20260904-0001/magcache_shift9_profile.json","profile_ratio_count":50,"profile_sha256":"1fab27b88c5fd74a96de85c4e9f4cd29809289c2cd7df2cb0e143b1b3bef64b7","profile_timestep_count":50,"retention_ratio":0.2,"threshold":0.12} | {"end_step":49,"interval":2,"layers":null,"start_step":10} |
| Method decision threshold | setting | N/A | 0.1 | 0.1 | 0.12 | N/A |
| Paired prompt/video count | count | 16 | 16 | 16 | 16 | 16 |
| Mean end-to-end generation latency | seconds | 198.118 | 100.179 | 100.342 | 82.623 | 157.306 |
| Mean generation latency excluding first request | seconds | 197.090 | 99.006 | 99.537 | 81.751 | 156.395 |
| Mean prompt-paired speedup versus exact baseline | x | 1.0000x | 1.9800x | 1.9756x | 2.3996x | 1.2595x |
| Paired speedup mean 95% CI lower bound | x | 1.0000x | 1.9535x | 1.9560x | 2.3776x | 1.2581x |
| Paired speedup mean 95% CI upper bound | x | 1.0000x | 2.0065x | 1.9952x | 2.4217x | 1.2609x |
| Whole-tail residual reuse ratio | ratio | 0.00% | 55.50% | 55.25% | 64.00% | 0.00% |
| Managed-attention reuse ratio | ratio | 0.00% | 0.00% | 0.00% | 0.00% | 38.00% |
| Rank-maximum method cache residency | MiB | N/A | N/A | N/A | 93.086 | 4469.250 |
| Rank-maximum peak CUDA allocation | GiB | 64.648 | 64.919 | 65.011 | 64.739 | 69.019 |
| Mean per-video relative latent L1 | ratio | 0 | 0.21220811 | 0.19017638 | 0.20628979 | 0.11202521 |
| Mean per-video relative latent L2 | ratio | 0 | 0.22196082 | 0.20029409 | 0.20910963 | 0.12026318 |
| Global final-latent mean squared error | latent^2 | 0 | 0.00096205773 | 0.00085714943 | 0.00086409876 | 0.00030731797 |
| Global final-latent root mean squared error | latent | 0 | 0.031017055 | 0.029277115 | 0.029395557 | 0.017530487 |
| Mean per-video latent cosine similarity | score | 1 | 0.972928 | 0.97676449 | 0.97642227 | 0.99201843 |
| Worst-pair latent maximum absolute error | latent | 0 | 1.380482 | 1.544873 | 1.0172707 | 0.88852233 |
| Global decoded RGB mean absolute error | RGB [0,1] | 0 | 0.037955453 | 0.033632861 | 0.035686105 | 0.021000863 |
| Mean per-video relative decoded RGB L1 | ratio | 0 | 0.10185231 | 0.090459731 | 0.0965513 | 0.057434255 |
| Mean per-video relative decoded RGB L2 | ratio | 0 | 0.14909101 | 0.13462862 | 0.14043425 | 0.086646952 |
| Global decoded RGB mean squared error | RGB^2 [0,1] | 0 | 0.0044393284 | 0.0038704309 | 0.0038256432 | 0.0015481153 |
| Global decoded RGB root mean squared error | RGB [0,1] | 0 | 0.066628286 | 0.062212787 | 0.061851784 | 0.039346097 |
| VBench subject consistency | score | 0.89380137 | 0.89702881 | 0.89432785 | 0.89475423 | 0.89688152 |
| VBench background consistency | score | 0.92789714 | 0.93021762 | 0.93540599 | 0.93450559 | 0.92873332 |
| VBench motion smoothness | score | 0.97972387 | 0.98251468 | 0.98049392 | 0.98417877 | 0.97955512 |
| VBench dynamic degree (descriptive motion-present rate) | score | 0.8125 | 0.6875 | 0.75 | 0.75 | 0.8125 |
| VBench aesthetic quality | score | 0.51230822 | 0.50728897 | 0.51357029 | 0.49625587 | 0.50919903 |
| VBench imaging quality | score | 0.57293649 | 0.54647656 | 0.5736181 | 0.52750633 | 0.57086181 |
| VideoScore2 visual quality | score | 3.125 | 3.1875 | 3.0625 | 3.375 | 3.0625 |
| VideoScore2 text alignment | score | 3.3125 | 3.1875 | 3.375 | 3.5 | 3.25 |
| VideoScore2 physical consistency | score | 3.375 | 3.3125 | 3.0625 | 3.4375 | 3.125 |
