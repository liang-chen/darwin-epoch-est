
class mGaussian:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

class ModelState:
    def __init__(self, mindex, epoch):
        self.epoch = epoch
        self.model = mindex
