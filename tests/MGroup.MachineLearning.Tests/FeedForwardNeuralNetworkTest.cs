using System;
using Xunit;
using static Tensorflow.KerasApi;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using MGroup.MachineLearning.Preprocessing;

namespace MGroup.MachineLearning.Tests
{
    public class FeedForwardNeuralNetworkTest
    {
        //learning the polynomial : f(x)=0.5x^3+2x^2+x in x -> [-3,0.5], f'(x)=1.5x^2+4x+1
        static double[,] trainX = { { -3 }, { -2.90000000000000 }, { -2.80000000000000 }, { -2.70000000000000 }, { -2.60000000000000 }, { -2.50000000000000 }, { -2.40000000000000 }, { -2.30000000000000 }, { -2.20000000000000 }, { -2.10000000000000 }, { -2 }, { -1.90000000000000 }, { -1.80000000000000 }, { -1.70000000000000 }, { -1.60000000000000 }, { -1.50000000000000 }, { -1.40000000000000 }, { -1.30000000000000 }, { -1.20000000000000 }, { -1.10000000000000 },
                { -1 }, { -0.900000000000000 }, { -0.800000000000000 }, { -0.700000000000000 }, { -0.600000000000000 }, { -0.500000000000000 }, { -0.400000000000000 }, { -0.300000000000000 }, { -0.200000000000000 }, { -0.100000000000000 }, { 0 }, { 0.100000000000000 }, { 0.200000000000000 }, { 0.300000000000000 }, { 0.400000000000000 }, { 0.500000000000000 } };

        static double[,] trainY = { { 1.50000000000000 }, { 1.72550000000000 }, { 1.90400000000000 }, { 2.03850000000000 }, { 2.13200000000000 }, { 2.18750000000000 }, { 2.20800000000000 }, { 2.19650000000000 }, { 2.15600000000000 }, { 2.08950000000000 }, { 2 }, { 1.89050000000000 }, { 1.76400000000000 }, { 1.62350000000000 }, { 1.47200000000000 }, { 1.31250000000000 }, { 1.14800000000000 }, { 0.981500000000000 }, { 0.816000000000000 }, { 0.654500000000000 }, { 0.500000000000000 },
                { 0.355500000000000 }, { 0.224000000000000 },{ 0.108500000000000 },{ 0.0120000000000000 }, { -0.0625000000000000 }, { -0.112000000000000 }, { -0.133500000000000 }, { -0.124000000000000 }, { -0.0805000000000001 },{ 0 },{ 0.120500000000000 },{ 0.284000000000000 },{ 0.493500000000000 },{ 0.752000000000000 },{ 1.06250000000000 } };

        static double[,] testX = { { -2.80000000000000 }, { -1.98000000000000 }, { -1.13000000000000 }, { 0.230000000000000 } };

        static double[,] testY = { { 1.90400000000000 }, { 1.979604000000000 }, { 0.702351500000000 }, { 0.341883500000001 } };

        static double[][,] testGradient = { new double[1, 1] { { 1.5600 } }, new double[1, 1] { { -1.0394 } }, new double[1, 1] { { -1.6046 } }, new double[1, 1] { { 1.9994 } } };

        [Fact]
        public static void MinMaxNormalizationWithAdam() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new MinMaxNormalization(), new MinMaxNormalization(),
                new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.005f),
                keras.losses.MeanSquaredError(), new[]
                {
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                },
                5000));

        [Fact (Skip = "Needs more epochs")]
        public static void MinMaxNormalizationWithSGD() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new MinMaxNormalization(), new MinMaxNormalization(),
                new TensorFlow.Keras.Optimizers.SGD(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.005f),
                keras.losses.MeanSquaredError(), new[]
                {
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                },
                5000));

        [Fact (Skip = "Needs more epochs")]
        public static void MinMaxNormalizationWithRMSProp() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new MinMaxNormalization(), new MinMaxNormalization(),
                new TensorFlow.Keras.Optimizers.RMSProp(new Tensorflow.Keras.ArgsDefinition.RMSpropArgs(), dataType: Tensorflow.TF_DataType.TF_DOUBLE),
                keras.losses.MeanSquaredError(), new[]
                {
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                },
                5000));

        [Fact (Skip = "Removed to save time during testing execution")]
        public static void ZScoreNormalizationWithAdam() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new ZScoreNormalization(), new ZScoreNormalization(),
                new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.005f),
                keras.losses.MeanSquaredError(), new[]
                {
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                },
                5000));

        [Fact(Skip = "Needs more epochs")]
        public static void ZScoreNormalizationWithSGD() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new ZScoreNormalization(), new ZScoreNormalization(),
                new TensorFlow.Keras.Optimizers.SGD(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.005f),
                keras.losses.MeanSquaredError(), new[]
                {
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                },
                5000));

        [Fact(Skip = "Needs more epochs")]
        public static void ZScoreNormalizationWithRMSProp() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new ZScoreNormalization(), new ZScoreNormalization(),
                new TensorFlow.Keras.Optimizers.RMSProp(new Tensorflow.Keras.ArgsDefinition.RMSpropArgs(), dataType: Tensorflow.TF_DataType.TF_DOUBLE),
                keras.losses.MeanSquaredError(), new[]
                {
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                    new NeuralNetworkLayerParameter(50, ActivationType.SoftMax),
                },
                5000));

        private static void TestFeedForwardNeuralNetwork(FeedForwardNeuralNetwork neuralNetwork)
        {
            neuralNetwork.Train(trainX, trainY);
            var responses = neuralNetwork.EvaluateResponses(testX);
            var gradients = neuralNetwork.EvaluateResponseGradients(testX);
            CheckResponseAccuracy(testY, responses);
            CheckResponseGradientAccuracy(testGradient, gradients);
        }

        private static void CheckResponseAccuracy(double[,] data, double[,] prediction)
        {
            var deviation = new double[prediction.GetLength(0), prediction.GetLength(1)];
            var norm = 0d;
            for (int i = 0; i < prediction.GetLength(0); i++)
            {
                for (int j = 0; j < prediction.GetLength(1); j++)
                {
                    deviation[i, j] = (data[i, j] - prediction[i, j]) / data[i, j];
                    norm += Math.Pow(deviation[i, j], 2);
                }
            }
            norm = Math.Sqrt(norm);
            Assert.True(norm < 0.03, $"Response norm was above threshold (norm value: {norm})");
        }

        private static void CheckResponseGradientAccuracy(double[][,] dataGradient, double[][,] gradient)
        {
            var norm = new double[gradient.GetLength(0)];
            var normTotal = 0d;
            for (int k = 0; k < gradient.GetLength(0); k++)
            {
                var deviation = new double[gradient[k].GetLength(0), gradient[k].GetLength(1)];
                for (int i = 0; i < gradient[k].GetLength(0); i++)
                {
                    for (int j = 0; j < gradient[k].GetLength(1); j++)
                    {
                        deviation[i, j] = (dataGradient[k][i, j] - gradient[k][i, j]) / dataGradient[k][i, j];
                        norm[k] += Math.Pow(deviation[i, j], 2);
                    }
                }
                normTotal += norm[k];
            }
            normTotal = Math.Sqrt(normTotal);
            Assert.True(normTotal < 0.1, $"Gradient norm was above threshold (norm value: {norm})");
        }
    }
}
