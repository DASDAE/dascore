import pickle


def save_pickle(self,filename):
    """
    quick and dirty save method
    """
    with open(filename,'wb') as f:
        pickle.dump(self,f)
    print('saved the class to: ' + filename)

def load_pickle(self,filename):
    """
    quick and dirty load method
    """
    # test if class data file exist
    with open(filename,'rb') as f:
        temp = pickle.load(f)
    self.__dict__.update(temp.__dict__)