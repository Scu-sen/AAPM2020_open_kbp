import numpy as np

class Interpolate:
    def __init__(self, ratio=(0.5, 0.5), axis=3):
        assert ratio[0] + ratio[1] == 1
        assert ratio[0] >= 0 and  ratio[1] >= 0
        assert axis in [1, 2, 3]
        self.r1, self.r2 = ratio
        self.backr = self.r1/self.r2
        self.axis = axis
        
    def forward(self, a):
        assert len(a.shape) == 4
        
        idx1 = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        flip = False
        if self.backr > 1:
            idx1[self.axis] = np.s_[::-1]
            a = a[tuple(idx1)]
            self.backr = 1. / self.backr
            flip = True
        
        catshape = list(a.shape)
        catshape[self.axis] = 1
        a = np.concatenate((np.zeros(catshape), a), axis=self.axis)
        
        idx1 = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idx1[self.axis] = np.s_[:-1]
        idx2 = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idx2[self.axis] = np.s_[1:]
        if flip:
            a = self.r2*a[tuple(idx1)] + self.r1*a[tuple(idx2)]
        else:
            a = self.r1*a[tuple(idx1)] + self.r2*a[tuple(idx2)]
        
        if flip:
            idx1[self.axis] = np.s_[::-1]
            a = a[tuple(idx1)]
            self.backr = 1. / self.backr
        return a
    
    def backward(self, a):
        assert len(a.shape) == 4
        idx1 = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idx2 = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        
        flip = False
        if self.backr > 1:
            idx1[self.axis] = np.s_[::-1]
            a = a[tuple(idx1)]
            self.backr = 1. / self.backr
            flip = True
        
        catshape = list(a.shape)
        catshape[self.axis] = 1
        a = np.concatenate((np.zeros(catshape), a), axis=self.axis)
        
        for i in range(a.shape[self.axis]-1):
            idx1[self.axis] = i+1
            idx2[self.axis] = i
            a[tuple(idx1)] -= a[tuple(idx2)]*self.backr
        
        idx2[self.axis] = np.s_[1:]
        a = a[tuple(idx2)]/self.r2
        if flip:
            idx2[self.axis] = np.s_[::-1]
            a = a[tuple(idx2)]
            a *= self.r2/self.r1
            self.backr = 1. / self.backr
        return a

if __name__=='__main__':
    a = np.random.rand(12,20,20,20)
    
    r1 = 0.75
    inter = Interpolate(ratio=(r1, 1-r1), axis=3)
    # print('a: ', a)
    b = inter.forward(a)
    # print('b: ', b)
    c = inter.backward(b)
    # print('c: ', c)
    np.testing.assert_almost_equal(a, c)
    
    r1 = 0.25
    inter = Interpolate(ratio=(r1, 1-r1), axis=3)
    # print('a: ', a)
    b = inter.forward(a)
    # print('b: ', b)
    c = inter.backward(b)
    # print('c: ', c)
    np.testing.assert_almost_equal(a, c)
    