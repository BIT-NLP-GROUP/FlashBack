from datasets import load_from_disk
from tqdm import tqdm
data_dir = '/home/16tb_hdd/lrh/downloads/datasets/wikitext-2-v1'


def main():
	dataset = load_from_disk(data_dir)
	train = dataset['train']
	with open("./train_titles.txt",'w') as fp:
		for data in tqdm(train):
			text = data['text'][1:-2]
			if len(text) == 0:
				continue
			if text[0] == '=' and text[-1] == '=':
				title = text[2:-2]
				if title[0] == '=' and title[-1] == '=':
					continue
				fp.write(title+'\n')

if __name__ == '__main__':
		main()
	


