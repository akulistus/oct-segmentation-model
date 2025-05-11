class Prediction:
    def __init__(self):
        self.table = [
            ["retinal_drusen ", "Возрастная макулярная дегенерация"],
            ["intraretinal_cyst", "Диабетический макулярный отёк"],
            ["subretinal_hyperreflective_material", "Хориоидальная неоваскуляризация (в т.ч. при ВМД)"],
            ["neuroepithelium_detachment", "Центральная серозная хориоретинопатия"],
            ["vitreomacular_traction", "Витреомакулярный тракционный синдром"],
            ["lamellar_macular_rupture", "Ламеллярная макулярная дыра"],
            ["vitreous_detachment", "Возрастные изменения, миопия высокой степени"]
        ]

    def find_matches(self, pred_labels, pred_scores):
        result = set([])
        for index, lable in enumerate(pred_labels):
            if (pred_scores[index] > 0.5):
                for row in self.table:
                    if lable == row[0]:
                        result.add(row[1])
        
        return result