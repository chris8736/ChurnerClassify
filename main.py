from src.classifiers import XGBoost, LightGBM, Bagging, GradientBoost, NeuralNet, RandomForest, SVM
from src.preprocess import Preprocess
from src.data import Data

from statistics import mean, stdev
from dataclasses import dataclass

@dataclass
class pp_score:
    pp: Preprocess
    mean: float
    std_dev: float

    def __repr__(self):
        return f"PP: {self.pp.__str__()}, mean: {self.mean}, stdev: {self.std_dev}"

def select_pp(cls, n=2, pps=[]):
    data = Data()
    results = []
    idx = 1
    total = n*len(pps)
    for pp in pps:
        scores = []
        for i in range(n):
            print(f"Iteration: {idx}/{total}")
            cls.pp = pp
            scores.extend(cls.cross_validate(data.training))
            idx += 1
        
        results.append(pp_score(pp, mean(scores), stdev(scores)))
    
    results.sort(key=lambda pp: pp.mean)
    return results


if __name__ == "__main__":
    results = select_pp(XGBoost(),pps=[
        Preprocess(),
        Preprocess().scale(),
        Preprocess().convert_to_pcs(),
        Preprocess().remove_correlated_features().remove_low_impact(),
        Preprocess().remove_correlated_features().remove_low_impact().reduce_features(["Total_Trans_Amt", "Total_Trans_Ct"], lambda x,y: x/y, output="Avg_Trans_Amt").remove_columns(["Total_Trans_Amt", "Total_Trans_Ct"])
    ])

    for r in results[:3]:
        print(r)