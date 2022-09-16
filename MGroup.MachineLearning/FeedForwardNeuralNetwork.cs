using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace MGroup.MachineLearning
{
    public class FeedForwardNeuralNetwork
    {
        string Preprocessing { get; set; }

        //Optimization Parameters
        public string Optimizer { get; set; }
        public double LearningRate { get; set; }
        public int BatchSize { get; set; }
        public int Epochs { get; set; }
        public int DisplayStep { get; set; }

        //Neural Network Architecture
        public int NumHiddenLayers { get; set;}
        public int[] NumNeuronsPerLayer { get; set; }
        public Activation[] ActivationFunctionPerLayer { get; set; }

        public FeedForwardNeuralNetwork()
        {
            InitializeParameters();
        }

        private void InitializeParameters()
        {
            Optimizer = "SGD";
            LearningRate = 0.01;
            BatchSize = 64;
            Epochs = 100;
            NumHiddenLayers = 1;
            NumNeuronsPerLayer = new int[] { 10 };
        }

        public void TrainNetwork()
        {
            tf.enable_eager_execution();

            var optimizer = keras.optimizers.SGD((float)LearningRate);
            
        }    

    }

}
