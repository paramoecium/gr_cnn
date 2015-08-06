import numpy as np
import math as m
import argparse
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d # Instance.signal_alignment()
from sklearn.utils import shuffle
import csv


def readFile(filepath):
	print 'Reading file: ', filepath 
	data = []
	with open(filepath, 'r') as fr:
		for line in fr:
			data.append([float(num)for num in line.split(',')])
	return data

def readDataset(filepath):
	data = np.array( readFile(filepath) )
	dataSet = [] # array of Instance
	index = data[:,0]
	last_i = 0
	for i, shift in enumerate(index):
		if shift == 0:
			if(last_i != i):
				dataSet.append( Instance(data[range(last_i, i),:]) )
			last_i = i
	
	dataSet.append( Instance(data[range(last_i, len(index)),:]) )
	return dataSet # return an array of Instance

def eigenvector_projection(instances):
	pca = PCA(n_components=1)
	pca.fit(np.concatenate(instances))
	return [np.concatenate(pca.transform(i)) for i in instances]


class Instance:
	def __init__(self,rawData):
		#print rawData
		self.timestamp = rawData[:,-1]
		self.timestamp = [ts-self.timestamp[0]for ts in self.timestamp] # relative time
		self.accel_1 = rawData[:,1]
		self.accel_2 = rawData[:,2]
		self.accel_3 = rawData[:,3]
		self.alpha_1 = rawData[:,4]
		self.alpha_2 = rawData[:,5]
		self.alpha_3 = rawData[:,6]
		self.label = -1 # -1 means unassigned

	def set_label(self, label):
		self.label = label

	def alignment(self, specifiedSize):
		self.accel_1 = self.__signal_alignment(self.accel_1, specifiedSize)
		self.accel_2 = self.__signal_alignment(self.accel_2, specifiedSize)
		self.accel_3 = self.__signal_alignment(self.accel_3, specifiedSize)
		self.alpha_1 = self.__signal_alignment(self.alpha_1, specifiedSize)
		self.alpha_2 = self.__signal_alignment(self.alpha_2, specifiedSize)
		self.alpha_3 = self.__signal_alignment(self.alpha_3, specifiedSize)
		self.timestamp = range(specifiedSize)

	def __signal_alignment(self, original_signal, specifiedSize):
		x = self.timestamp
		y = original_signal
		f = interp1d(x, y, kind='cubic')
		xnew = np.linspace(x[0], x[-1], specifiedSize)
		return f(xnew)

	def __cart2sph(self,x,y,z):						# to spherical coordinates
		XsqPlusYsq = x**2 + y**2
		r = m.sqrt(XsqPlusYsq + z**2)				# r
		elev = m.atan2(z,m.sqrt(XsqPlusYsq))		# theta
		az = m.atan2(y,x)							# phi
		return [r, elev, az]

	def accelToSph(self):
		accel_sph = []
		for a_xyz in self.get_accel():
			accel_sph.append(self.__cart2sph(a_xyz[0],a_xyz[1],a_xyz[2]))
		return accel_sph

	def alphaToSph(self):
		alpha_sph = []
		for a_xyz in self.get_alpha():
			alpha_sph.append(self.__cart2sph(a_xyz[0],a_xyz[1],a_xyz[2]))
		return alpha_sph

	def accel_abs(self):
		accel = [0]*self.get_length()
		for i in range(self.get_length()):
			accel[i] = (self.accel_1[i]**2 + self.accel_2[i]**2 + self.accel_3[i]**2)**0.5
		return accel
	
	def alpha_abs(self):
		alpha = [0]*self.get_length()
		for i in range(self.get_length()):
			alpha[i] = (self.alpha_1[i]**2 + self.alpha_2[i]**2 + self.alpha_3[i]**2)**0.5
		return alpha

	def get_length(self):
		return len(self.timestamp)

	def get_accel(self):
		return [list(accel) for accel in zip(self.accel_1, self.accel_2, self.accel_3)]

	def get_alpha(self):
		return [list(alpha) for alpha in zip(self.alpha_1, self.alpha_2, self.alpha_3)]

	def get_timestamp(self):
		return self.timestamp

	def get_timespan(self):
		return self.timestamp[-1]-self.timestamp[0]

	def toCSVarray(self, instanceId):
		csvArray = []
		accel_sph = self.accelToSph()
		alpha_sph = self.alphaToSph()
		for i in range(self.get_length()):
			csvArray.append(\
			[instanceId, self.label, self.accel_1[i], self.accel_2[i], self.accel_3[i], self.alpha_1[i], self.alpha_2[i], self.alpha_3[i]])
			#[instanceId, self.label]+accel_sph[i]+alpha_sph[i] )
		return csvArray

TRAIN_SET_RATIO = 0.8
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	#argparser.add_argument('fileName', type=str, help='fileName')
	args = argparser.parse_args()
	args = vars(args)
	dataSet = []
	label = []
	for i, fileName in enumerate( ['./data/arthur.csv', './data/brian.csv', './data/nofar.csv', './data/shalom.csv'] ):
		tmp = readDataset(fileName) # array of Instance
		tmp = [instance for instance in tmp if instance.get_timespan()>2] # filter out data shoter than 2 seconds
		for instance in tmp:
			instance.set_label(i+1) # label starts form 1, say 1,2,3,4
			instance.alignment(100)
		dataSet = dataSet + tmp
		label = label + [i]*len(tmp)
	dataSet, label = shuffle(dataSet, label, random_state=0)
	cutIndex = int(TRAIN_SET_RATIO*len(dataSet))
	for cutSet, setName in [(dataSet[:cutIndex], 'train'), (dataSet[cutIndex:], 'test')]:
		csvDataSet = []
		for id, instance in enumerate(cutSet):
			csvDataSet = csvDataSet + instance.toCSVarray(id)
		f = open("alignedData_"+setName+".csv","w")
		w = csv.writer(f)
		w.writerows(csvDataSet)
		f.close()

