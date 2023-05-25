from env import *
class KalmanFilter:
    def __init__(self, noise_velocity, noise_position) -> None:
        # Complete this function to construct

        # Assume that nothing is known 
        # about the state of the target at this instance

        self.Q = numpy.diag([noise_velocity, noise_velocity, noise_velocity,
                             noise_position, noise_position, noise_position])
        self.R = numpy.diag([noise_position, noise_position, noise_position])
        self.P = numpy.diag([self.R[0, 0], self.R[0, 0], self.R[0, 0],
                         self.Q[0, 0], self.Q[0, 0], self.Q[0, 0]])
        self.H = numpy.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.K = numpy.zeros((6, 3))
        self.state = State(Vector3D(), Vector3D())
        
        
        pass

    def input(self, observed_state:State, accel:numpy.ndarray, justUpdated:bool):

        # This function is executed multiple times during the reading.
        # When an observation is read, the `justUpdated` is true, otherwise it is false
        
        # accel is the acceleration(control) vector. 
        # It is dynamically obtained regardless of the state of the RADAR 
        # (i.e regardless of `justUpdated`) 

        # When `justUpdated` is false, the state is the same as the previously provided state
        # (i.e, given `observed_state` is not updated, it's the same as the previous outdated one)


        # Complete this function where current estimate of target is updated
 
        xhat = numpy.array([observed_state.position, observed_state.velocity]).reshape(-1, 1)
        u = accel.reshape(-1, 1)
        I = numpy.eye(6)

        if justUpdated:
           self.P = numpy.diag([self.R[0, 0], self.R[0, 0], self.R[0, 0],
                             self.Q[0, 0], self.Q[0, 0], self.Q[0, 0]])

        else:
        # prediction step
           A = numpy.eye(6)
           A[0:3, 3:6] = I[0:3, 0:3]
        
           if accel.shape == (3,):
              u = numpy.vstack((accel.reshape(3, 1), numpy.zeros((3,1))))

           else:
              u = accel.reshape(-1, 1)
              
           z = numpy.vstack((numpy.zeros((3,1)), self.state.position.reshape(-1,1)))
        
           xhatminus = numpy.dot(A, z) + u
           Pminus = numpy.dot(numpy.dot(A, self.P), A.T) + self.Q

           # update step
           K = numpy.dot(numpy.dot(Pminus, self.H.T), numpy.linalg.inv(numpy.dot(numpy.dot(self.H, Pminus), self.H.T) + self.R))
           xhat = xhatminus + numpy.dot(K, (observed_state.position.reshape(-1, 1) - numpy.dot(self.H, xhatminus)))
           self.P = numpy.dot((I - numpy.dot(K, self.H)), Pminus)

           # update state
           self.state = State(xhat[0:3, :].reshape(3), xhat[3:6, :].reshape(3))
    
        pass


    def get_current_estimate(self)->State:
        
        # Complete this function where the current state of the target is returned
        return self.state
        
        pass
