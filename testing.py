import time
from animated_logger import progress_bar, graph_all_mats, graph_mat

for i in range(500):
    progress_bar(i/500)
    time.sleep(0.1)
print("\n")

data = {
    "loss": [0.555,0.44,0.22,0.33,0.10,0.01],
    "accuracy": [0.10,0.20,0.40,0.7,1,1.1]
}

graph_all_mats(data,len(data["loss"]))