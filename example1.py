import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2*np.random.random((3,1)) - 1
    
    def sigmoid_func(self,x):
        a = 1/(1+np.exp(-x))
        return a

    def derivative_sigmoid(self,x):
        a = x*(1-x)
        return a
    
    def predict(self,input):
        input = input.astype(float)
        p = np.dot(input,self.synaptic_weights)
        r_output = self.sigmoid_func(p)
        return r_output

    
    def train(self,input,output,interation):
        print(neuralnetwork.synaptic_weights)

        for i in range(interation):
            # r_input = input.astype(float)
            # # print(neuralnetwork.synaptic_weights)

            # p = np.dot(r_input,self.synaptic_weights)
            # # print(neuralnetwork.synaptic_weights)

            r_output = self.predict(input)
            print(r_output)
            print("output")
            # output =output.transpose
            print(output)

            e = output - r_output
            print("error")
            print(e)
            delta = e*self.derivative_sigmoid(output)
            weight_change = []
            weight_change = np.dot(input.T , e * self.derivative_sigmoid(r_output))
            print(weight_change)
            self.synaptic_weights =  self.synaptic_weights + weight_change


if __name__ == "__main__":
    neuralnetwork = NeuralNetwork()

    print("Random initial synaptic weights:")
    print(neuralnetwork.synaptic_weights)

    input = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    
    output = np.array([0,1,1,0])[np.newaxis]
    output = output.T
    
    neuralnetwork.train(input,output,1)
    #1d new weights, error resolved
    print("New synaptic weights:")
    print(neuralnetwork.synaptic_weights)

    #ndaray error below
    # one = str(input("Enter first input"))
    # two = str(input("Enter second input"))
    # third = str(input("Enter third input"))

    # inputs = np.array([input_one,input_two,input_third])
    # print(neuralnetwork.predict(inputs))




