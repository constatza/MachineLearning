using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
{
    // for each double[] => rows: input quantity, columns: input dimension
    public interface INeuralNetwork
	{
        IEnumerable<NeuralNetworkLayerParameter> NeuralNetworkLayerParameters { get; }

        // TrainX: input (stimulus)
        // TrainY: output (response)
        // TestX and TestY, to be removed
        void Train(double[,] trainX, double[,] trainY, double[,] testX, double[,] testY);

        double[,] Predict(double[,] data);

        double[][,] Gradient(double[,] data);

        void SaveNetwork(string netPath, string weightsPath);

        void LoadNetwork(string netPath, string weightsPath);
    }
}
