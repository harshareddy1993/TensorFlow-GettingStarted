# ML Overview
In traditional software development, the input and algorithm are known and the output function is written to produce an output. In Machine Learning, a set of input and output data is given to the program for figuring out the algorithm.
## Neural Network
A neural network is a stack of layers where each layer consists of a level of predefined math and internal variables. The input is fed to the neural network, which is passed along to the stack of layers.
Training a neural network is the process of letting the layers repeatedly try to map the input to output. This is achieved by tuning the internal variables until the network learns to produce the output given the inputs.
The training process is performed for thousands or even millions of internal variables for figuring out the relationship.
## Training Process
Below example is a general process for any ML program written in TensorFlow.

    l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) 
    model = tf.keras.Sequential([l0])
    model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
    history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
    model.predict([100.0])
The entire training process happening as part of the fit function is about tuning the internal variables (weights) of the networks to the best possible values, so that they can map input to output. This is achieved through an optimization process called Gradient Descent, which iteratively adjusts parameters, nudging them in the corredct direction a bit at a time until the besr values are reached. 
The function that measuers how bad or goodth e model is doing during each iteration **loss function** and the goal of each nudge is to minimize the loss function.
The training process starts with a forward pass, where input data is fed into the neural network. The model applies its internal math on the input and uses weights to predict an answer.
Once a value is predicted, the difference between the predicted value and the correct value is calculated using the loss funciton.
After determining the loss, internal variables of all layers of the neural network are adjusted, so that the loss is minimized. 

![Forward Pass](https://video.udacity-data.com/topher/2019/March/5c7f0b37_tensorflow-l2f2/tensorflow-l2f2.png)
![Back Propagation](https://video.udacity-data.com/topher/2019/March/5c7f0ba2_tensorflow-l2f3/tensorflow-l2f3.png)
