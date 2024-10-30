import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        max_depth=100
        wrong=False
        for _ in range(max_depth):
            wrong=False
            for x,y in dataset.iterate_once(1):
                if self.get_prediction(x)!=nn.as_scalar(y):
                    wrong=True
                    self.w.update(x,nn.as_scalar(y))
            if wrong==False:
                break
                

            


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden1_w=nn.Parameter(1,512)
        self.hidden1_b=nn.Parameter(1,512)
        self.hidden2_w=nn.Parameter(512,512)
        self.hidden2_b=nn.Parameter(1,512)
        self.output_w=nn.Parameter(512,1)
        self.output_b=nn.Parameter(1,1)
        
        
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        h1=nn.ReLU(nn.AddBias(nn.Linear(x,self.hidden1_w),self.hidden1_b))
        h2=nn.ReLU(nn.AddBias(nn.Linear(h1,self.hidden2_w),self.hidden2_b))
        output=nn.AddBias(nn.Linear(h2,self.output_w),self.output_b)
        return output
        
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        
        result=self.run(x)
        return nn.SquareLoss(result,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learn_rate=-0.05
        batch_size=200
        while True:
            for x,y in dataset.iterate_once(batch_size):
                loss=self.get_loss(x,y)
                grad_1w,grad_1b,grad_2w,grad_2b,grad_outw,grad_outb=nn.gradients(loss,[self.hidden1_w,self.hidden1_b,self.hidden2_w,self.hidden2_b,self.output_w,self.output_b])   
                self.hidden1_w.update(grad_1w,learn_rate)
                self.hidden1_b.update(grad_1b,learn_rate)
                self.hidden2_w.update(grad_2w,learn_rate)
                self.hidden2_b.update(grad_2b,learn_rate)
                self.output_w.update(grad_outw,learn_rate)
                self.output_b.update(grad_outb,learn_rate)
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(loss)<0.02:
                break
        
                
            


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden1_w=nn.Parameter(784,200)
        self.hidden1_b=nn.Parameter(1,200)
        self.hidden2_w=nn.Parameter(200,200)
        self.hidden2_b=nn.Parameter(1,200)
        self.output_w=nn.Parameter(200,10)
        self.output_b=nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        
        h1=nn.ReLU(nn.AddBias(nn.Linear(x,self.hidden1_w),self.hidden1_b))
        h2=nn.ReLU(nn.AddBias(nn.Linear(h1,self.hidden2_w),self.hidden2_b))
        output=nn.AddBias(nn.Linear(h2,self.output_w),self.output_b)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        result=self.run(x)
        return nn.SoftmaxLoss(result,y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learn_rate=-0.5
        batch_size=100
        while True:
            for x,y in dataset.iterate_once(batch_size):
                loss=self.get_loss(x,y)
                grad_1w,grad_1b,grad_2w,grad_2b,grad_outw,grad_outb=nn.gradients(loss,[self.hidden1_w,self.hidden1_b,self.hidden2_w,self.hidden2_b,self.output_w,self.output_b])   
                self.hidden1_w.update(grad_1w,learn_rate)
                self.hidden1_b.update(grad_1b,learn_rate)
                self.hidden2_w.update(grad_2w,learn_rate)
                self.hidden2_b.update(grad_2b,learn_rate)
                self.output_w.update(grad_outw,learn_rate)
                self.output_b.update(grad_outb,learn_rate)
            if dataset.get_validation_accuracy()>0.98:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_size = 200

        self.W_ih = nn.Parameter(self.num_chars, hidden_size)
        self.W_hh = nn.Parameter(hidden_size, hidden_size)
        self.W_ho = nn.Parameter(hidden_size, len(self.languages))
        self.b_h = nn.Parameter(1, hidden_size)
        self.b_o = nn.Parameter(1, len(self.languages))
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.W_ih), self.b_h))
        for x in xs[1:]:
            h = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(x, self.W_ih), nn.Linear(h, self.W_hh)), self.b_h))
        return nn.AddBias(nn.Linear(h, self.W_ho), self.b_o)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted = self.run(xs)
        return nn.SoftmaxLoss(predicted, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.01
        batch_size = 100
        
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.W_ih, self.W_hh, self.W_ho, self.b_h, self.b_o])
                
                self.W_ih.update(gradients[0], -learning_rate)
                self.W_hh.update(gradients[1], -learning_rate)
                self.W_ho.update(gradients[2], -learning_rate)
                self.b_h.update(gradients[3], -learning_rate)
                self.b_o.update(gradients[4], -learning_rate)
            
            accuracy = dataset.get_validation_accuracy()
            if accuracy > 0.87:
                break
