from sklearn.cluster import KMeans
import os, numpy
from sklearn.feature_extraction.text import TfidfVectorizer

class Cluster:

    def __init__(self):
        self.train_file = os.path.join('data', 'sample')

    def run_main(self):
        self.load_data()
        self.vectorize()
        self.train()

    def load_data(self):
        self.training_data = []
        with open(self.train_file, 'r') as fd:
            for line in fd.readlines():
                self.training_data.append(line)

    def vectorize(self):
        vect = TfidfVectorizer(stop_words='english')  
        self.X = vect.fit_transform(self.training_data)

    def train(self):
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
        kmeans.fit(self.X)        
        import pdb;pdb.set_trace()


if __name__ == "__main__":
    clus_obj = Cluster()
    clus_obj.run_main()
