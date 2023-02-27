import numpy as np
from rescomp import optimizer as rcopt

class NoisySoftRobot(rcopt.System):
    """Add random noise to the training data."""
    def __init__(self, train_time, test_time, dt, n_copies, U_noise_mag, D_noise_mag):
        super().__init__('softrobot_noisy', train_time, test_time, dt, 6, 6, is_diffeq=False, is_driven=True)
        self.orig = rcopt.SoftRobotSystem(train_time, test_time, dt)
        self.n_copies = n_copies
        self.U_noise_mag = U_noise_mag
        self.D_noise_mag = D_noise_mag
    
    def add_noise(self, array, val):
        return array + np.random.normal(scale=val, size=array.shape)
    
    def get_train_test_data(self, cont_test=True):
        tr, (Utr, Dtr), (ts, Dts), Uts = self.orig.get_train_test_data(cont_test)
        #Add noise
        return (
            [tr] * self.n_copies,
            (
                [self.add_noise(Utr, self.U_noise_mag) for _ in range(self.n_copies)],
                [self.add_noise(Dtr, self.D_noise_mag) for _ in range(self.n_copies)]
            ),
            (ts, Dts), Uts
        )
    
    def get_random_test(self):
        return self.orig.get_random_test()
