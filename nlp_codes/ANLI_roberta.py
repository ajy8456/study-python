import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

import torch
import numpy as np
import os
import random
import argparse
import copy
import pickle
import ipdb
import logging 
from tqdm import tqdm

import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import (DataLoader, Dataset)
from torch.nn import CrossEntropyLoss

from transformers import RobertaTokenizer, RobertaForSequenceClassification

import NLP_utils

class ANLIDataset_tokenized(torch.utils.data.Dataset):
	def __init__(self, anli_data, max_length):
		super(ANLIDataset_tokenized, self).__init__()
		self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
		self.max_length = max_length

		self.train_data = anli_data
		self.data_length = len(anli_data)
		self.features = []
		self.build_dataset()

	def __getitem__(self, index):
		
		return self.features[index]

	def __len__(self):
		return len(self.features)

	def build_dataset(self):
		for idx in range(self.data_length):
			featured_datum = self.convert_data_to_feature(self.train_data[idx])
			self.features.append(featured_datum)

	def convert_data_to_feature(self, data):
		context_ids = self.tokenizer(data['context'])['input_ids']
		hypothesis_ids = self.tokenizer(data['hypothesis'])['input_ids']
		
		label_id = NLP_utils.convert_label(data['label'])

		context_ids, hypothesis_ids = NLP_utils.truncation(context_ids, hypothesis_ids, self.max_length)
		
		# [2] is </s>-sep token
		input_ids = context_ids + [2] + hypothesis_ids[1:]
		token_type_ids = [0] *len(context_ids) + [1] * len(hypothesis_ids)
	  
		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)
		
		# Zero-pad up to the sequence length.
		padding = [0] * (self.max_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		token_type_ids += padding  
		if not len(input_ids) == self.max_length:
			ipdb.set_trace()
		# assert len(input_ids) == self.max_length
		# assert len(input_mask) == self.max_length
		# assert len(token_type_ids) == self.max_length
		
		feature = {'input_ids':input_ids, 'input_mask': input_mask, 'token_type_ids': token_type_ids, 'label_id': label_id}
		return feature

parser = argparse.ArgumentParser(description='train roberta using ANLI task')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument("--data_path", type=str, help="data folder path where anli data saved")
parser.add_argument("--difficulty", type=str, default='all',choices=['R1','R2','R3','all'], help='choose the version (R1, R2, R3) to want to use')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=30)

parser.add_argument('--temp_scale', action='store_true')
parser.add_argument('--do_train', action='store_true', help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")

parser.add_argument("--ckpt", type=str, help='The model ckpt for fine-tuning or test')

parser.add_argument("--result_dir", type=str,
						help="The output directory where the model checkpoints will be written.")
parser.add_argument('--save_name', type=str, default='debugging')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True

def _train(args, model, train_dataloader, optimizer, device):

	tr_loss = 0
	model.train() #added by HSY 8/21/2022 13:12:00
	for batch_idx, batch in enumerate(tqdm(train_dataloader)):
		batch = NLP_utils.processing_dataloader(batch)
		batch = NLP_utils.dict_to_device(batch, device)

		output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['input_mask'], labels=batch['label_id'])
		loss = output.loss 

		loss.backward()
		tr_loss += loss.item()

		optimizer.step()
		optimizer.zero_grad()
			
	return model, optimizer


def _eval(args, model, eval_dataloader, device):
	model.eval()
	eval_loss = 0
	eval_accuracy = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
			batch = NLP_utils.processing_dataloader(batch)
			batch = NLP_utils.dict_to_device(batch, device)

			output = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['input_mask'], labels=batch['label_id'])
			loss = output.loss
			eval_loss += loss.item()
			eval_accuracy += NLP_utils.correct_or_not(output, batch['label_id'])
		eval_accuracy = eval_accuracy/len(eval_dataloader.dataset)
	return	eval_loss, eval_accuracy

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not args.do_train and not args.do_test:
		raise ValueError("At least one of `do_train` or `do_test` must be True.")

	args.output_dir = os.path.join(args.result_dir, str(args.lamda) + '/' + str(args.beta))
	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
		print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
	else:
		print("Making Output directory ({})".format(args.output_dir))
		os.makedirs(args.output_dir, exist_ok=True)
		

	tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
	model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=3) # label: entailment, contradiction, neutral
	
	# Update config to finetune token type embeddings
	model.config.type_vocab_size = 2 

	# Create a new Embeddings layer, with 2 possible segments IDs instead of 1
	model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, model.config.hidden_size)
					
	# Initialize it
	model.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
	model.to(device)

	# Prepare optimizer
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.beta},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	optimizer = AdamW(optimizer_grouped_parameters, lr= args.lr)
	

	# prepare dataset
	if args.do_train:
		#open log file 
		log_file_name = os.path.join(args.output_dir, args.save_name+'.log')
		logging.basicConfig(filename=log_file_name,
						format='%(asctime)s %(message)s', 
						filemode='w') 
		
		logger=logging.getLogger() 
		logger=logging.getLogger() 
		logger.setLevel(logging.DEBUG)
		logger.info(args)

		print('Training ANLI Roberta Model...') #
		logger.info('Training ANLI Roberta Model...') 
		

		print('Loading data + preprocessing...') #
		logger.info('Loading data + preprocessing...') 
		sys.stdout.flush() #
		
	# processing train, val, test datasets and for 'all' arg, merging three datasets (R1, R2, R3)	
	if args.difficulty == 'all':
		data_list = ['R1', 'R2', 'R3']
		train_data = []
		eval_data = []
		test_data = []
		for dataset in data_list:
			data_path = os.path.join(args.data_path, dataset)
			train_data += NLP_utils.load_jsonl_list(os.path.join(data_path, 'train.jsonl'))
			eval_data += NLP_utils.load_jsonl_list(os.path.join(data_path, 'dev.jsonl'))
			test_data += NLP_utils.load_jsonl_list(os.path.join(data_path, 'test.jsonl'))
	else:
		current_data_path = os.path.join(args.data_path, args.difficulty)

		train_data_path = os.path.join(current_data_path, 'train.jsonl')
		train_data = NLP_utils.load_jsonl_list(train_data_path)

		eval_data_path = os.path.join(current_data_path, 'dev.jsonl')
		eval_data = NLP_utils.load_jsonl_list(eval_data_path)
		
		test_data_path = os.path.join(current_data_path, 'test.jsonl')
		test_data = NLP_utils.load_jsonl_list(test_data_path)
	
	train_dataset = ANLIDataset_tokenized(train_data, args.max_seq_length)
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	
	eval_dataset = ANLIDataset_tokenized(eval_data, args.max_seq_length)
	eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
	
	test_dataset = ANLIDataset_tokenized(test_data, args.max_seq_length)
	test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size)
	print('Finished loading data')
	logger.info('Finished loading data') 
	sys.stdout.flush() #

	best_accuracy = 0
	save_version = 0

	# start training 
	if args.do_train:
		for epoch in range(args.epochs):
			print('Training start at epoch {}...'.format(epoch))
			logger.info('Training start at epoch {}...'.format(epoch))
			sys.stdout.flush() 

			model, optimizer = _train(args, model, train_dataloader, optimizer, device)
			
			if args.do_eval:
				eval_loss, eval_accuracy = _eval(args, model, eval_dataloader, device)
			
			print('epoch:  {}   Dev loss:  {}   Dev accuracy:  {}'.format(epoch, eval_loss, eval_accuracy))
			logger.info('epoch:  {}   Dev loss:  {}   Dev accuracy:  {}'.format(epoch, eval_loss, eval_accuracy))
			sys.stdout.flush() 

			# save best model only (if for the same performance, save as different version)
			if best_accuracy < eval_accuracy:
				print('updating best model at epoch {}...'.format(epoch))
				logger.info('updating best model at epoch {}...'.format(epoch)) 
				sys.stdout.flush() 
				best_accuracy = eval_accuracy
				save_path = os.path.join(args.output_dir, args.save_name+'_best_model.pt')
				NLP_utils.model_save(model, optimizer, save_path)

			elif best_accuracy == eval_accuracy:
				print('saving diffent version of  best model at epoch {}...'.format(epoch))
				logger.info('saving diffent version of best model at epoch {}...'.format(epoch)) 
				sys.stdout.flush() 
				save_path = os.path.join(args.output_dir, args.save_name+'_version_'+str(save_version)+'_epoch_'+str(epoch)+'.pt')
				NLP_utils.model_save(model, optimizer, save_path)
				save_version += 1

	if args.do_test:
		if not args.ckpt:
			raise ValueError("You need to add ckpt for test")
		
		model.load_state_dict(torch.load(args.ckpt)['model'])

		test_loss, test_accuracy = _eval(args, model, test_dataloader, device)
		print('Test loss:  {}   Test accuracy:  {}'.format(test_loss, test_accuracy))


if __name__ == "__main__":
	main()
