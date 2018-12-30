import numpy as np

class NeuralNetwork():
    def __init__(self,x,y):
        self.input = x
        self.y = y
        self.output = np.zeros(self.y.shape)
        np.random.seed(1)
        self.synaptic_weights = 2*np.random.random((3,1)) - 1
    
    def sigmoid_func(self,x):
        a = 1/(1+np.exp(-x))
        return a

    def derivative_sigmoid(self,x):
        a = x*(1-x)
        return a
    
    def ff(self,input):
        self.input = input.astype(float)
        return self.sigmoid_func(np.dot(input,self.synaptic_weights))
    

    
    def train(self,interation):
        print(neuralnetwork.synaptic_weights)

        for i in range(interation):
            # r_input = input.astype(float)
            # # print(neuralnetwork.synaptic_weights)

            # p = np.dot(r_input,self.synaptic_weights)
            # # print(neuralnetwork.synaptic_weights)

            self.output = self.ff(self.input)
            print(self.output)
            print("output")
            # output =output.transpose
            print(output)

            e = self.y - self.output
            print("error")
            print(e)
            # delta = e*self.derivative_sigmoid(output)

            weight_change = np.dot(input.T , e * self.derivative_sigmoid(self.output))
            # print(weight_change)
            self.synaptic_weights =  self.synaptic_weights + weight_change


if __name__ == "__main__":
    input = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    
    output = np.array([0,1,1,0])[np.newaxis]
    output = output.T
    neuralnetwork = NeuralNetwork(input,output)

    print("Random initial synaptic weights:")
    print(neuralnetwork.synaptic_weights)

    
    
    neuralnetwork.train(1000)
    #1d new weights, error resolved
    print("New synaptic weights:")
    print(neuralnetwork.synaptic_weights)

    #ndaray error below
    # one = str(input("Enter first input"))
    # two = str(input("Enter second input"))
    # third = str(input("Enter third input"))

    # inputs = np.array([input_one,input_two,input_third])
    print("output")
    print(neuralnetwork.ff(input))



