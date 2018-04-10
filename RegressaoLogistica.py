import numpy as np

def RegresaoLogistica(w, y, x, b : float, alfa : float, n : int):
	for i in range(3):
		print("\nIteração {}:\n".format(i + 1))
		
		z = np.dot(w.T, x) + b 
		z = z.sum(axis = 0)
		
		print("Z:")
		print(z)
		
		#calcula sigmoíde
		a = 1/(1 + np.exp(-z))
		print("A:")
		print(a)
	
		#função de custo
		j = (1/n) * (np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a))))

		print("J:")
		print(j)

		#descida gradiente
		dz = a - y
		print("DZ:")
		print(dz)
		dw = (1/n) * x * dz.T
		dw = dw.sum(axis = 0)
		print("DW")
		print(dw)
		db = (1/n) * np.sum(dz)
		w = w - alfa * dw
		print("W")
		print(w)
		b = b - alfa * db
		print("b")
		print(b)


W = np.array([[2, 3]])
Y = np.array([[0, 1]])
X = np.array([[4, 5]])

RegresaoLogistica(W, Y, X, 3, 0.003, 2)
