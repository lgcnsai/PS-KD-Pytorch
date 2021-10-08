import numpy as np


def metric_ece_aurc_eaurc(confidences, truths, bin_size=0.1):
    
    confidences = np.asarray(confidences)
    truths = np.asarray(truths)
     
    total = len(confidences)
    predictions = np.argmax(confidences, axis=1)
    max_confs = np.amax(confidences, axis=1)
     
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    accs = []
    avg_confs = []
    bin_counts = []
    ces = []
    

    for upper_bound in upper_bounds:
        lower_bound = upper_bound - bin_size
        acc, avg_conf, bin_count = compute_bin(lower_bound, upper_bound, max_confs, predictions, truths)
        accs.append(acc)
        avg_confs.append(avg_conf)
        bin_counts.append(bin_count)
        ces.append(abs(acc - avg_conf) * bin_count)



    ece = 100 * sum(ces) / total

    aurc, e_aurc = calc_aurc(confidences, truths)

    return ece, aurc * 1000, e_aurc * 1000
     

def compute_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
        accuracy = float(correct)/len(filtered_tuples)
        bin_count = len(filtered_tuples)
        return accuracy, avg_conf, bin_count



def calc_aurc(confidences, labels):
    confidences = np.array(confidences)
    labels = np.array(labels)
    predictions = np.argmax(confidences, axis=1)
    max_confs = np.max(confidences, axis=1)
            
    n = len(labels)
    indices = np.argsort(max_confs)
    labels, predictions, confidences = labels[indices][::-1], predictions[indices][::-1], confidences[indices][::-1]
    risk_cov = np.divide(np.cumsum(labels != predictions).astype(np.float), np.arange(1, n+1))
    nrisk = np.sum(labels != predictions)
    aurc = np.mean(risk_cov)
    opt_aurc = (1./n) * np.sum(np.divide(np.arange(1, nrisk + 1).astype(np.float), n - nrisk + np.arange(1, nrisk + 1)))
    eaurc = aurc - opt_aurc
            
    return aurc, eaurc