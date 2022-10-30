using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
{
    public enum ActivationType
    {
        NotSet,
        Linear,
        RelU,
        Sigmoid,
        TanH,
        SoftMax
    }

    public class NeuralNetworkLayerParameter
    {
        public int Neurons { get; }
        public ActivationType ActivationType { get; }

        public NeuralNetworkLayerParameter(int neurons, ActivationType activationType)
        {
            this.Neurons = neurons;
            this.ActivationType = activationType;
        }
    }
}
