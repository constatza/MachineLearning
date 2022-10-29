using System;
using Xunit;
using static Tensorflow.KerasApi;
using MGroup.MachineLearning.NeuralNetworks;

namespace MGroup.MachineLearning.Tests
{
    public class FeedForwardNeuralNetworkTest
    {
        [Fact]
        public static void RunTest()
        {
            //learning the polynomial : 0.5x^3+2x^2+x in x -> [-3,0.5]

            double[,] trainX = { { -3 }, { -2.90000000000000 }, { -2.80000000000000 }, { -2.70000000000000 }, { -2.60000000000000 }, { -2.50000000000000 }, { -2.40000000000000 }, { -2.30000000000000 }, { -2.20000000000000 }, { -2.10000000000000 }, { -2 }, { -1.90000000000000 }, { -1.80000000000000 }, { -1.70000000000000 }, { -1.60000000000000 }, { -1.50000000000000 }, { -1.40000000000000 }, { -1.30000000000000 }, { -1.20000000000000 }, { -1.10000000000000 }, 
                { -1 }, { -0.900000000000000 }, { -0.800000000000000 }, { -0.700000000000000 }, { -0.600000000000000 }, { -0.500000000000000 }, { -0.400000000000000 }, { -0.300000000000000 }, { -0.200000000000000 }, { -0.100000000000000 }, { 0 }, { 0.100000000000000 }, { 0.200000000000000 }, { 0.300000000000000 }, { 0.400000000000000 }, { 0.500000000000000 } };

            double[,] trainY = { { 1.50000000000000 }, { 1.72550000000000 }, { 1.90400000000000 }, { 2.03850000000000 }, { 2.13200000000000 }, { 2.18750000000000 }, { 2.20800000000000 }, { 2.19650000000000 }, { 2.15600000000000 }, { 2.08950000000000 }, { 2 }, { 1.89050000000000 }, { 1.76400000000000 }, { 1.62350000000000 }, { 1.47200000000000 }, { 1.31250000000000 }, { 1.14800000000000 }, { 0.981500000000000 }, { 0.816000000000000 }, { 0.654500000000000 }, { 0.500000000000000 }, 
                { 0.355500000000000 }, { 0.224000000000000 },{ 0.108500000000000 },{ 0.0120000000000000 }, { -0.0625000000000000 }, { -0.112000000000000 }, { -0.133500000000000 }, { -0.124000000000000 }, { -0.0805000000000001 },{ 0 },{ 0.120500000000000 },{ 0.284000000000000 },{ 0.493500000000000 },{ 0.752000000000000 },{ 1.06250000000000 } };

            double[,] testX = { { -2.80000000000000 }, {- 1.98000000000000 }, { -1.13000000000000 }, { 0.230000000000000 } };

            double[,] testY = { { 1.90400000000000 }, { 1.979604000000000 },{ 0.702351500000000 }, { 0.341883500000001 } };

            //MinMaxNormalization normalization = new MinMaxNormalization();

            var neuralNetwork = new FeedForwardNeuralNetwork()
            {               
                NumHiddenLayers = 2,
                NumNeuronsPerLayer = new int[] { 50, 50},
                Epochs = 5000,
                Optimizer = new Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.005f),
                LossFunction = keras.losses.MeanSquaredError(),
                ActivationFunctionPerLayer = new string[] { "softmax", "softmax" },
                //Normalization = new MinMaxNormalization(),
            };

            neuralNetwork.Train(trainX, trainY);

            var prediction = neuralNetwork.Predict(testX);

            // var gradient = neuralNetwork.Gradient(testX);

            CheckAccuracy(testY, prediction);
        }

        private static void CheckAccuracy(double[,] data,double[,] prediction)
        {
            var deviation = new double[prediction.GetLength(0), prediction.GetLength(1)];
            var norm = 0d;
            for (int i = 0; i < prediction.GetLength(0); i++)
            {
                for (int j = 0; j < prediction.GetLength(1); j++)
                {
                    deviation[i, j] = (data[i, j] - prediction[i, j]) / data[i,j];
                    norm += Math.Pow(deviation[i, j],2);
                }
            }           
            norm = Math.Sqrt(norm);
            Assert.True(norm < 0.03, $"Norm was above threshold (norm value: {norm})");
        }
    }
}
