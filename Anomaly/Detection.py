

class Detection:
    def __init__(self, predicted_values, modified_values, index_of_modified_values, pe_mean, pe_std, h):
        self.tp = 0 #true_positives
        self.tn = 0 #true_negatives
        self.fp = 0 #false_positives
        self.fn = 0 #false_negatives
        self.specificity = 0 #tpr
        self.sensitivity = 0 #tnr
        self.fpr = 0
        self.fnr = 0
        self.detect_anomalies(predicted_values, modified_values, index_of_modified_values, pe_mean, pe_std, h)

    def detect_anomalies(self, predicted_values, modified_values, index_of_modified_values, pe_mean, pe_std, h):
        window_max = pe_mean + h * pe_std
        window_min = pe_mean - h * pe_std
        error_list = [abs((predictions - modified)/predictions) for predictions, modified in zip(predicted_values, modified_values)]
        for i in error_list:
            if not window_min < i < window_max:
                if error_list.index(i) in index_of_modified_values:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if error_list.index(i) in index_of_modified_values:
                    self.fn += 1
                else:
                    self.tn += 1
        # print(f"TP: {self.tp}, FN: {self.fn}")
        # print(f"FP: {self.fp}, TN: {self.tn}")
        self.specificity = self.tn/(self.tn + self.fp)
        self.sensitivity = self.tp/(self.tp + self.fn)



