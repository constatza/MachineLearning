using System;
using System.Linq;
using Xunit;
using Tensorflow.Keras;
using static Tensorflow.KerasApi;

namespace MGroup.MachineLearning.Tests
{
    public class FeedForwardNeuralNetworkLinearRegression
    {
        [Fact]
        private static void RunTest()
        {
            double[,] trainX = {{3.3f }, {4.4f }, {5.5f }, {6.71f }, {6.93f }, {4.168f }, {9.779f }, {6.182f },
                    { 7.59f }, {2.167f }, {7.042f }, {10.791f }, {5.313f }, {7.997f }, {5.654f }, {9.27f }, {3.1f }} ;

            //double[,] trainX = {{1f }, {2f }, {3f }, {4f }, {5f }, {6f }, {7f }, {8f },
            //        { 9f }, {10f } };

            //double[,] trainX = { { 1, 0 }, { 1, 1 }, { 0, 0 }, { 0, 1 } };

            double[,] trainY = { {1.7f}, {2.76f}, {2.09f}, {3.19f}, {1.694f}, {1.573f}, {3.366f},
                    {2.596f}, {2.53f}, {1.221f}, {2.827f}, {3.465f}, {1.65f}, {2.904f}, {2.42f}, {2.94f}, {1.3f} } ;

            //double[,] trainY = {{2f }, {4f }, {6f }, {8f }, {10f }, {12f }, {14f }, {16f },
            //        { 18f },{ 20f } };

            //double[,] trainY = { { 1 }, { 0 }, { 0 }, { 1 } };

            double[,] data = { { 5.5f } };

            //float[,] data = { { 1, 1 }, { 0, 0 }, { 0, 1 }, { 0, 1 } };

            MinMaxNormalization normalization = new MinMaxNormalization();

            var neuralNetwork = new FeedForwardNeuralNetwork()
            {               
                NumHiddenLayers = 1,
                //NumNeuronsPerLayer = new int[] { 1, 1, 1, 1, 1, 1, 1, 1 },
                NumNeuronsPerLayer = new int[] { 1 },
                //LearningRate = 0.01,
                //BatchSize = 17,
                Epochs = 1000,
                Optimizer = keras.optimizers.Adam(0.005f),
                LossFunction = keras.losses.MeanAbsoluteError(),
                //ActivationFunctionPerLayer = new Activation[] { keras.activations.Relu, keras.activations.Relu, keras.activations.Relu, keras.activations.Relu, keras.activations.Relu, keras.activations.Relu, keras.activations.Relu, keras.activations.Relu },
                ActivationFunctionPerLayer = new Activation[] { keras.activations.Linear },
                Normalization = new MinMaxNormalization(),
            };

            neuralNetwork.Train(trainX, trainY);

            neuralNetwork.Predict(data);

            neuralNetwork.Gradient(data);
        }
    }
}
