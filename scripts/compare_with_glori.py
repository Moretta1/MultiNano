# generate the comparison of predicted methylation rate and corresponding GLORI rate

import numpy as np

print("compare with GLORI process")
# load real glori methylation rate:
glori_pth = '/data/home/grp-lizy/wangrulan/tandem/data/GLORI/HEK293T/HEK293T_allAs_cov20.bed'
glori_dict = {}
with open(glori_pth, 'r') as file:
    content = file.read()
    content_split = content.split("\n")
    for i in range(0, len(content_split)-1):
        line = content_split[i].rstrip()
        items = line.split("\t")
        info = items[3]
        key_val = info.split("|")[0]+'|'+info.split("|")[1]

        ratio = 0 if np.mean(float(items[4]) + float(items[10])) < 0.1 else 1
        glori_dict[key_val] = ratio


# predict result
predict_file = '/data/home/grp-lizy/wangrulan/tandem/result/predict_m6A.tsv' 
predict_dict = []

# generate comparison with GLORI file
# contains: chromosome, position, methylation rate at certain position
generated_file = '/data/home/grp-lizy/wangrulan/tandem/result/predict_glori_comparison.tsv'
f1 = open(generated_file, 'w')

with open(predict_file, 'r') as pred_file:
    content = pred_file.read()
    content_split = content.split("\n")
    for i in range(0, len(content_split) - 1):
        line = content_split[i].rstrip()
        info = line.split('\t')
        chr = info[0]
        position = info[1]
        motif = info[2]
        pred_label = info[4]

        glori_key = chr+'|'+position
        glori_label = glori_dict[glori_key]
        print(pred_label, glori_label, sep='\t', file=f1)

f1.close()











