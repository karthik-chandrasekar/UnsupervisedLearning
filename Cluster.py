from sklearn.cluster import KMeans, MiniBatchKMeans
import os, numpy
from sklearn.feature_extraction.text import TfidfVectorizer

class Cluster:

    def __init__(self):
        self.train_file = os.path.join('data', 'sample')

    def run_main(self):
        self.load_data()
        self.vectorize()

        #KMeans - K++
        print "KMeans - K++"
        self.kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10000)
        self.train()
        self.get_metrics()

        #MiniBatchKMeans - K++
        print "MiniBatchKMeans - K++"
        self.kmeans = MiniBatchKMeans(n_clusters=3, init='k-means++', n_init=10000)       
        self.train()
        self.get_metrics()
 
        #KMeans - Random
        print "KMeans - Random"
        self.kmeans = KMeans(n_clusters=3, init='random', n_init=10000)
        self.train()
        self.get_metrics()

        #MiniBatchKMeans - K++
        print "MiniBatchKMeans - Random"
        self.kmeans = MiniBatchKMeans(n_clusters=3, init='random', n_init=10000)       
        self.train()
        self.get_metrics()


    def load_data(self):
        self.training_data = []
        with open(self.train_file, 'r') as fd:
            for line in fd.readlines():
                self.training_data.append(line)

    def vectorize(self):
        self.vect = TfidfVectorizer(stop_words='english')  
        self.X = self.vect.fit_transform(self.training_data)

    def train(self):
        self.kmeans.fit(self.X)        

    def get_metrics(self):
        print self.kmeans.labels_ 

    def test(self):
        self.test_data = ["I know both Ashok and Harini"]
        self.Y = self.vect.fit_transform(self.test_data)
        print self.kmeans.predict(self.Y)


if __name__ == "__main__":
    clus_obj = Cluster()
    clus_obj.run_main()
