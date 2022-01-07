import numpy as np

class MF():

    def __init__(self, R, K, alpha, beta, iterations):

        """
        Melaakukan faktor matriks untuk memprediksi entri kosong dalam matriks.
        
        Arguments

        - R (ndarray)   : user-item rating matrix/ rating dari pengguna

        - K (int)       : number of latent dimensions/jumlah dimensi

        - alpha (float) : learning rate/ tingkat pembelajaran

        - beta (float)  : regularization parameter/ parameter reguler
        """
        

        self.R = R

        self.num_users, self.num_items = R.shape

        self.K = K

        self.alpha = alpha

        self.beta = beta
        self.iterations = iterations


    def train(self):

        # Pengguna awal dan item fitur matriks

        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))

        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        

        # Inisialisasi bias

        self.b_u = np.zeros(self.num_users)

        self.b_i = np.zeros(self.num_items)

        self.b = np.mean(self.R[np.where(self.R != 0)])
        

        # Buat daftar sampel pelatihan

        self.samples = [

            (i, j, self.R[i, j])

            for i in range(self.num_users)

            for j in range(self.num_items)

            if self.R[i, j] > 0

        ]
        

        # Lakukan stapsiss tray keturunan atas sejumlah iterasi

        training_process = []

        for i in range(self.iterations):

            np.random.shuffle(self.samples)

            self.sgd()
            mse = self.mse()

            training_process.append((i, mse))

            if (i+1) % 10 == 0:

                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        

        return training_process


    def mse(self):
        
        """
        Fungsi untuk menghitung total kesalahan mean persegi 
        """

        xs, ys = self.R.nonzero()

        predicted = self.full_matrix()

        error = 0

        for x, y in zip(xs, ys):

            error += pow(self.R[x, y] - predicted[x, y], 2)

        return np.sqrt(error)


    def sgd(self):
        
        """
        Lakukan stokastik keturunan gradien
        """

        for i, j, r in self.samples:

            # Prediksi komputer dan kesalahan

            prediction = self.get_rating(i, j)

            e = (r - prediction)
            

            # Update biases

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])

            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            

            # Buat salinan baris P karena kami perlu memperbaruinya tetapi gunakan nilai yang lama untuk pembaruan Q

            P_i = self.P[i, :][:]
            

            # update user dan item fitur matriks

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])

            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j,:])


    def get_rating(self, i, j):
        
        """
        Get prediksi rating dari pengguna i dan j
        """

        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)

        return prediction
    

    def full_matrix(self):
        
        """
        Komputer full matriks menggunakan hasil dari bias, P dan Q
        """

        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)

R = np.array([

[5, 3, 0, 1],

[4, 0, 0, 1],

[1, 1, 0, 5],

[1, 0, 0, 4],

[0, 1, 5, 4],

])

mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)

training_process = mf.train()
print()

print("P x Q:")

print(mf.full_matrix())
print()

print("Global bias:")

print(mf.b)
print()

print("User bias:")

print(mf.b_u)
print()

print("Item bias:")

print(mf.b_i)


