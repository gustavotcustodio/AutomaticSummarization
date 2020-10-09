import os

for filename in os.listdir('./rouge'):
    fullfilename = './rouge/%s' % filename
    with open(fullfilename) as f:
        parts = f.read().split('\n\n')
        label1 = [int(sent.split(']')[0][7:])
                  for sent in parts[2].split('\n')[:-1]]
        n_sents = len(parts[1].split('\n'))
        label_list = [1 if i+1 in label1 else 0 for i in range(n_sents)]
        f.close()

    with open('./labels/%s' % filename, 'w') as f:
        f.write('\n'.join(str(l) for l in label_list))
        f.close()
    print('Labels salvos com sucesso')
