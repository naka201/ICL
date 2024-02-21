import numpy as np

results_dict = {
   "Tokyo": 0.18280917,
    "Paris": 0.17461455,
    "Madrid": 0.1353072,
    "Rome": 0.11710726,
    "Berlin": 0.24445519,
    "sister": 0.31148055,
    "father": 0.27853718,
    "stepmother": 0.18487892,
    "wife": 0.23504342,
    "niece": 0.22490789,
    "walked": 0.18943454,
    "said": 0.4196937,
    "forgot": 0.3440456,
    "won": 0.24728437,
    "heard": 0.31572634,
    "mice": 0.29405016,
    "media": 0.39672092,
    "phenomena": 0.27654707,
    "teeth": 0.210121,
    "knives": 0.21946561,
    "action": 0.40652612,
    "addition": 0.37835813,
    "satisfaction": 0.2905448,
    "suggestion": 0.6933979,
    "information": 0.73793936
}


# 結果を降順にソート
sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)

# 上位5つの結果を表示
top_results = sorted_results[:5]
for word, similarity in top_results:
    print(f"{word} : {similarity}")
