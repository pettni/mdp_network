import scipy.sparse as sp
import numpy as np

from best.utils import sparse_tensordot

def test_sparse1():

	a = sp.coo_matrix(np.random.randn(5, 9))
	b = np.random.randn(9)

	c = sparse_tensordot(a, b, 0)

	np.testing.assert_equal(c, a.dot(b))

def test_sparse2():

	a = sp.coo_matrix(np.random.randn(5, 8))
	b = np.random.randn(9,8,7)

	c = sparse_tensordot(a, b, 1)

	np.testing.assert_equal(c.shape, (9,5,7))

	for i in range(9):
		for j in range(7):
			np.testing.assert_equal(c[i,:,j], a.dot(b[i,:,j]))

def test_sparse3():

	a = sp.coo_matrix(np.random.randn(5, 7))
	b = np.random.randn(9,8,7)

	c = sparse_tensordot(a, b, 2)

	np.testing.assert_equal(c.shape, (9,8,5))

	for i in range(9):
		for j in range(8):
			np.testing.assert_equal(c[i,j,:], a.dot(b[i,j,:]))


def test_sparse4():

	a = sp.coo_matrix(np.random.randn(5, 9))
	b = np.random.randn(9,8,7)

	c = sparse_tensordot(a, b, 0)

	np.testing.assert_equal(c.shape, (5,8,7))

	for i in range(8):
		for j in range(7):
			np.testing.assert_equal(c[:,i,j], a.dot(b[:,i,j]))
