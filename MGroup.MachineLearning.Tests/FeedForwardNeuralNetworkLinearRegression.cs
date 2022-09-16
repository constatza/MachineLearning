using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using Xunit;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace MGroup.MachineLearning.Tests
{
    public class FeedForwardNeuralNetworkLinearRegression
    {
        private static void RunTest()
        {
            double[] trainX = {3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f,
                    7.59f, 2.167f, 7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f };

            double[] trainY = { 1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f,
                    2.596f, 2.53f, 1.221f, 2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f };

            double[] testX = { };

            double[] testY = { };

            var neuralNetwork = new FeedForwardNeuralNetwork();

            neuralNetwork.NumHiddenLayers = 1;
            neuralNetwork.NumNeuronsPerLayer = new int[] { 10 };
            neuralNetwork.Optimizer = "SGD";
            neuralNetwork.LearningRate = 0.01;
            neuralNetwork.BatchSize = 64;
            

            Activation[] activationFunctionPerLayer = { keras.activations.Linear };

            float learning_rate = 0.1f;
            int display_step = 100;
            int batch_size = 256;
            int training_steps = 1000;
        }
    }
}
