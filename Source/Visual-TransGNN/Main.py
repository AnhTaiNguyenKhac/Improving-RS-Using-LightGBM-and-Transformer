# import torch as t
# import Utils.TimeLogger as logger
# from Utils.TimeLogger import log
# from Params import args
# from Model import TransGNN
# from DataHandler import DataHandler
# import numpy as np
# import pickle
# from Utils.Utils import *
# import os
# import setproctitle

# class Coach:
# 	def __init__(self, handler):
# 		self.handler = handler

# 		print('USER', args.user, 'ITEM', args.item)
# 		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
# 		self.metrics = dict()
# 		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
# 		for met in mets:
# 			self.metrics['Train' + met] = list()
# 			self.metrics['Test' + met] = list()

# 	def makePrint(self, name, ep, reses, save):
# 		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
# 		for metric in reses:
# 			val = reses[metric]
# 			ret += '%s = %.4f, ' % (metric, val)
# 			tem = name + metric
# 			if save and tem in self.metrics:
# 				self.metrics[tem].append(val)
# 		ret = ret[:-2] + '  '
# 		return ret

# 	def run(self):
# 		self.prepareModel()
# 		log('Model Prepared')
# 		if args.load_model != None:
# 			self.loadModel()
# 			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
# 		else:
# 			stloc = 0
# 			log('Model Initialized')
# 		for ep in range(stloc, args.epoch):
# 			tstFlag = (ep % args.tstEpoch == 0)
# 			reses = self.trainEpoch()
# 			log(self.makePrint('Train', ep, reses, tstFlag))
# 			if tstFlag:
# 				reses = self.testEpoch()
# 				log(self.makePrint('Test', ep, reses, tstFlag))
# 				self.saveHistory()
# 			print()
# 		reses = self.testEpoch()
# 		log(self.makePrint('Test', args.epoch, reses, True))
# 		self.saveHistory()

# 	def prepareModel(self):
# 		self.model = TransGNN().cuda()
# 		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
	
# 	def trainEpoch(self):
# 		trnLoader = self.handler.trnLoader
# 		trnLoader.dataset.negSampling()
# 		epLoss, epPreLoss = 0, 0
# 		steps = trnLoader.dataset.__len__() // args.batch
# 		for i, tem in enumerate(trnLoader):
# 			ancs, poss, negs = tem
# 			ancs = ancs.long().cuda()
# 			poss = poss.long().cuda()
# 			negs = negs.long().cuda()

# 			bprLoss = self.model.calcLosses(ancs, poss, negs, self.handler.torchBiAdj)
# 			loss = bprLoss 
	
# 			epLoss += loss.item()
# 			epPreLoss += bprLoss.item()
# 			self.opt.zero_grad()
# 			loss.backward()
# 			self.opt.step()
# 			log('Step %d/%d: loss = %.3f         ' % (i, steps, loss), save=False, oneline=True)
# 		ret = dict()
# 		ret['Loss'] = epLoss / steps
# 		ret['preLoss'] = epPreLoss / steps
# 		return ret

# 	def testEpoch(self):
# 		tstLoader = self.handler.tstLoader
# 		epLoss, epRecall, epNdcg = [0] * 3
# 		i = 0
# 		num = tstLoader.dataset.__len__()
# 		steps = num // args.tstBat
# 		for usr, trnMask in tstLoader:
# 			i += 1
# 			usr = usr.long().cuda()
# 			trnMask = trnMask.cuda()
# 			usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)

# 			allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
# 			_, topLocs = t.topk(allPreds, args.topk)
# 			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
# 			epRecall += recall
# 			epNdcg += ndcg
# 			log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
# 		ret = dict()
# 		ret['Recall'] = epRecall / num
# 		ret['NDCG'] = epNdcg / num
# 		return ret

# 	def calcRes(self, topLocs, tstLocs, batIds):
# 		assert topLocs.shape[0] == len(batIds)
# 		allRecall = allNdcg = 0
# 		recallBig = 0
# 		ndcgBig =0
# 		for i in range(len(batIds)):
# 			temTopLocs = list(topLocs[i])
# 			temTstLocs = tstLocs[batIds[i]]
# 			tstNum = len(temTstLocs)
# 			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
# 			recall = dcg = 0
# 			for val in temTstLocs:
# 				if val in temTopLocs:
# 					recall += 1
# 					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
# 			recall = recall / tstNum
# 			ndcg = dcg / maxDcg
# 			allRecall += recall
# 			allNdcg += ndcg
# 		return allRecall, allNdcg

# 	def saveHistory(self):
# 		if args.epoch == 0:
# 			return
# 		with open('../History/' + args.save_path + '.his', 'wb') as fs:
# 			pickle.dump(self.metrics, fs)

# 		content = {
# 			'model': self.model,
# 		}
# 		t.save(content, '../Models/' + args.save_path + '.mod')
# 		log('Model Saved: %s' % args.save_path)

# 	def loadModel(self):
# 		ckp = t.load('../Models/' + args.load_model + '.mod')
# 		self.model = ckp['model']
# 		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

# 		with open('../History/' + args.load_model + '.his', 'rb') as fs:
# 			self.metrics = pickle.load(fs)
# 		log('Model Loaded')	

# if __name__ == '__main__':
# 	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# 	setproctitle.setproctitle('proc_title')
# 	logger.saveDefault = True
	
# 	log('Start')
# 	handler = DataHandler()
# 	handler.LoadData()
# 	log('Load Data')
# 	# exit(0)
# 	coach = Coach(handler)
# 	coach.run()

import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import TransGNN
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import setproctitle

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG', 'MRR', 'HR']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model is not None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			log('Model Initialized')

		early_stopped = False  

		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))

			if tstFlag:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, tstFlag))
				self.saveHistory()

				stop_ndcg = self.checkEarlyStopping(reses['NDCG'], metric='NDCG', mode='max')
				stop_recall = self.checkEarlyStopping(reses['Recall'], metric='Recall', mode='max')

				if stop_ndcg or stop_recall:
					log('Early stopping triggered.')
					early_stopped = True 
					break

			print()

		if not early_stopped:
			reses = self.testEpoch()
			log(self.makePrint('Test', args.epoch, reses, True))
			self.saveHistory()



	def prepareModel(self):
		self.model = TransGNN().cuda()
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
	
	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epPreLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch
		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			bprLoss = self.model.calcLosses(ancs, poss, negs, self.handler.torchBiAdj)
			loss = bprLoss 
	
			epLoss += loss.item()
			epPreLoss += bprLoss.item()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			log('Step %d/%d: loss = %.3f         ' % (i, steps, loss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['preLoss'] = epPreLoss / steps
		return ret

	# def testEpoch(self):
	# 	tstLoader = self.handler.tstLoader
	# 	epLoss, epRecall, epNdcg = [0] * 3
	# 	i = 0
	# 	num = tstLoader.dataset.__len__()
	# 	steps = num // args.tstBat
	# 	for usr, trnMask in tstLoader:
	# 		i += 1
	# 		usr = usr.long().cuda()
	# 		trnMask = trnMask.cuda()
	# 		usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)

	# 		allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
	# 		_, topLocs = t.topk(allPreds, args.topk)
	# 		recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
	# 		epRecall += recall
	# 		epNdcg += ndcg
	# 		log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
	# 	ret = dict()
	# 	ret['Recall'] = epRecall / num
	# 	ret['NDCG'] = epNdcg / num
	# 	return ret
 
	# def testEpoch(self):
	# 	tstLoader = self.handler.tstLoader
	# 	epLoss, epRecall, epNdcg = [0] * 3
	# 	i = 0
	# 	num = tstLoader.dataset.__len__()
	# 	steps = num // args.tstBat
		
	# 	for usr, trnMask in tstLoader:
	# 		i += 1
	# 		usr = usr.long().cuda()
	# 		trnMask = trnMask.cuda()
	# 		usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)

	# 		# Tạo mask động thay vì dùng mask toàn cục
	# 		# allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
	# 		allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0))
	# 		allPreds[trnMask.bool()] = -1e8  # Áp dụng mask tại đây
			
	# 		_, topLocs = t.topk(allPreds, args.topk)
	# 		recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
	# 		epRecall += recall
	# 		epNdcg += ndcg
	# 		log('Steps %d/%d: recall = %.2f, ndcg = %.2f' % (i, steps, recall, ndcg), save=False, oneline=True)
		
	# 	ret = dict()
	# 	ret['Recall'] = epRecall / num
	# 	ret['NDCG'] = epNdcg / num
	# 	return ret
 
 
	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epMrr, epHr = [0] * 4
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)

			allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = t.topk(allPreds, args.topk)

			recall, ndcg, mrr, hr = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epMrr += mrr
			epHr += hr
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f, mrr = %.2f, hr = %.2f' % (i, steps, recall, ndcg, mrr, hr), save=False, oneline=True)

		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['MRR'] = epMrr / num
		ret['HR'] = epHr / num
		return ret


	def checkEarlyStopping(self, val, metric='NDCG', mode='max'):
		'''
		Theo dõi giá trị metric để kiểm tra điều kiện dừng sớm.

		val: Giá trị của metric hiện tại (ví dụ NDCG ở epoch này)
		metric: Tên metric, ví dụ 'NDCG' hoặc 'Recall'
		mode: 'max' nếu cần tăng, 'min' nếu cần giảm

		Returns:
			True nếu nên dừng với metric này.
		'''
		if not hasattr(self, 'early_stop_stats'):
			self.early_stop_stats = {}
			self.patience = args.patience if hasattr(args, 'patience') else 5

		if metric not in self.early_stop_stats:
			best_val = -np.inf if mode == 'max' else np.inf
			self.early_stop_stats[metric] = {
				'best': best_val,
				'bad_epochs': 0,
				'mode': mode
			}
		

		stat = self.early_stop_stats[metric]
		improved = (val > stat['best']) if stat['mode'] == 'max' else (val < stat['best'])

		if improved:
			stat['best'] = val
			stat['bad_epochs'] = 0
		else:
			stat['bad_epochs'] += 1

		log(f"{metric}: current={val:.4f}, best={stat['best']:.4f}, bad_epochs={stat['bad_epochs']}/{self.patience}", save=False)

		return stat['bad_epochs'] >= self.patience


	# def calcRes(self, topLocs, tstLocs, batIds):
	# 	assert topLocs.shape[0] == len(batIds)
	# 	allRecall = allNdcg = 0
	# 	recallBig = 0
	# 	ndcgBig =0
	# 	for i in range(len(batIds)):
	# 		temTopLocs = list(topLocs[i])
	# 		temTstLocs = tstLocs[batIds[i]]
	# 		tstNum = len(temTstLocs)
	# 		maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
	# 		recall = dcg = 0
	# 		for val in temTstLocs:
	# 			if val in temTopLocs:
	# 				recall += 1
	# 				dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
	# 		recall = recall / tstNum
	# 		ndcg = dcg / maxDcg
	# 		allRecall += recall
	# 		allNdcg += ndcg
	# 	return allRecall, allNdcg
	
	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allMrr = allHr = 0

		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)

			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = mrr = hit = 0

			for val in temTstLocs:
				if val in temTopLocs:
					pos = temTopLocs.index(val)
					recall += 1
					dcg += 1 / np.log2(pos + 2)
					mrr += 1 / (pos + 1)
					hit = 1  # At least one hit

			recall /= tstNum
			ndcg = dcg / maxDcg if maxDcg > 0 else 0
			mrr /= tstNum  # average MRR per user

			allRecall += recall
			allNdcg += ndcg
			allMrr += mrr
			allHr += hit

		return allRecall, allNdcg, allMrr, allHr


	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('../History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		content = {
			'model': self.model,
		}
		t.save(content, '../Models/' + args.save_path + '.mod')
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		ckp = t.load('../Models/' + args.load_model + '.mod')
		self.model = ckp['model']
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		with open('../History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	setproctitle.setproctitle('proc_title')
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')
	# exit(0)
	coach = Coach(handler)
	coach.run()