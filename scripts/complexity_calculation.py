"""
| Article                        | Script                            | Meaning                                                     |
| ------------------------------ | --------------------------------- | ------------------------------------------------------------|
| (m_i)                          | model / names_model[model]        | the specific model                                          |
| (M)                            | 5                                 | number of models                                            |
| (\nu_i)                        | NT[model]                         | number of model functions                                   |
| (\nu_{0,i})                    | UT[model]                         | number of functions unique to the model                     |
| (r_{i,j})                      | rrt[model][j]                     | rate functions                                              |
| (\frac{\nu_{0,i}}{\nu_i})      | UT[model] / NT[model]             | uniqueness of the model                                     |
| (1-\frac{\sum r_{i,j}}{\nu_i}) | 1 - sum(rrt[model]) / NT[model]   | the degree of non-proliferation of functions                |
| (\frac{\nu_i}{\max\nu_i})      | NT[model] / max_NT                | the normalized relative complexity of the model by its size |
"""

import numpy as np 
from typing import Dict

def main() ->  Dict:

    GT = 10

    rrt_FHN = (3, 5, 3)
    rrt_LV = (5, 3, 3)
    rrt_p53 = (5, 3, 3, 3)
    rrt_WC = (3, 5, 3, 3, 3, 1)
    rrt_ML = (3, 5, 1, 1, 1, 1)
    rrt = (rrt_FHN, rrt_LV, rrt_p53, rrt_WC, rrt_ML)
    rrt = tuple(map(lambda item: tuple(np.array(item) / GT), rrt))

    NT_FHN = 3
    NT_LV = 3
    NT_p53 = 4
    NT_WC = 6
    NT_ML = 6

    UT_FHN = 0
    UT_LV = 0
    UT_p53 = 0
    UT_WC = 1
    UT_ML = 4

    max_NT = max([len(i) for i in rrt])

    UT = (UT_FHN, UT_LV, UT_p53, UT_WC, UT_ML)
    NT = (NT_FHN, NT_LV, NT_p53, NT_WC, NT_ML)

    alpha, beta, gamma = 1, 1, 1

    names_model = ("FHN", 
                "LV",
                "p53",
                "WC",
                "ML")

    Compl = []

    for model in range(5):

        complexity = alpha * (UT[model] / NT[model]) + beta * (1 - sum(rrt[model]) / NT[model]) + gamma * (NT[model] / max_NT)

        Compl.append([names_model[model], float(complexity)])
        

    complexity = dict(Compl)

    return complexity


