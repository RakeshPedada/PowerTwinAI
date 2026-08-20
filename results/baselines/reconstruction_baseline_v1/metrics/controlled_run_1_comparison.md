\# PowerTwinAI Reconstruction Baseline v1

\## Controlled Reproducibility Run 1



\## Purpose



This run was executed as an isolated reproducibility test.



The reconstruction used:



\- The preserved 67-image candidate baseline input set.

\- Python 3.10.10.

\- COLMAP 4.1.0.dev0.

\- COLMAP commit 5b76f53.

\- CUDA enabled.

\- The same recorded dense reconstruction parameters.



The controlled run used an isolated workspace and did not overwrite the normal PowerTwinAI reconstruction workspace.



\---



\## Historical Baseline v1



Input Images: 67



Registered Images: 67



Sparse Points: 15,934



Raw COLMAP Dense Points: 20,133



PowerTwinAI Cleaned Points: 19,417



Mesh Vertices: 24,788



Mesh Triangles: 49,008



\---



\## Controlled Reproducibility Run 1



Input Images: 67



Registered Images: 67



Sparse Points: 16,034



Raw COLMAP Dense Points: 28,096



\---



\## Comparison



| Metric | Historical Baseline v1 | Controlled Run 1 | Difference |

|---|---:|---:|---:|

| Input Images | 67 | 67 | 0 |

| Registered Images | 67 | 67 | 0 |

| Sparse Points | 15,934 | 16,034 | +100 |

| Raw Dense Points | 20,133 | 28,096 | +7,963 |



\---



\## Interpretation



The controlled reconstruction completed successfully and registered all 67 images.



Sparse reconstruction was close to the historical baseline, with 100 additional sparse points.



The raw dense point count differed significantly from the historical recorded baseline.



Therefore, Controlled Reproducibility Run 1 is recorded as a successful isolated reconstruction run, but not as an exact numerical reproduction of the historical Baseline v1.



The historical Baseline v1 remains unchanged.



No reconstruction algorithm, formula, threshold, or COLMAP parameter has been modified as part of this comparison.



Further investigation may be performed later to determine the cause of the numerical difference.



\## Status



CONTROLLED REPRODUCIBILITY RUN 1 RECORDED

