import os


def main():
    filelist = os.listdir('./rouge')
    for filename in filelist:

        with open(os.path.join('./rouge', filename)) as f:
            block_sententes = f.read().split('\n\n')[1]
            sents = [sent.split(' - ')[1]
                     for sent in block_sententes.split('\n')]
            f.close()

        with open(os.path.join('./sentences', filename[:-4]), 'w') as f:
            f.write("\n".join(sents))
            f.close()


if __name__ == "__main__":
    main()
