using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Losses;

namespace MGroup.MachineLearning
{
    public class FeedForwardNeuralNetwork2 : Model
    {
        string Preprocessing { get; set; }

        //Optimization Parameters
        public OptimizerV2 Optimizer { get; set; }
        public ILossFunc LossFunction { get; set; }
        public double LearningRate { get; set; }
        public int BatchSize { get; set; }
        public int Epochs { get; set; }
        public int DisplayStep { get; set; }
        public Layer[] Layer { get; private set; }

        //Neural Network Architecture
        public int NumHiddenLayers { get; set; }
        public int[] NumNeuronsPerLayer { get; set; }
        public Activation[] ActivationFunctionPerLayer { get; set; }

        Model model;
        NDArray trainX;
        NDArray testX;
        NDArray trainY;
        NDArray testY;

        public FeedForwardNeuralNetwork2(ModelArgs? args = null) : base(args: new ModelArgs())
        {
            //InitializeParameters();
            Preprocessing = "minmax";
            Optimizer = keras.optimizers.Adam();
            LearningRate = 0.001;
            BatchSize = -1;
            Epochs = 1;
            DisplayStep = 100;
            NumHiddenLayers = 1;
            NumNeuronsPerLayer = new int[] { 1 };
            ActivationFunctionPerLayer = new Activation[] { keras.activations.Linear };
            Layer = new Layer[1] { keras.layers.Dense(NumNeuronsPerLayer[0], ActivationFunctionPerLayer[0]) };
            //Args = args;
        }

        //private void InitializeParameters()
        //{
        //    Optimizer = "SGD";
        //    LearningRate = 0.01;
        //    BatchSize = 64;
        //    Epochs = 100;
        //    NumHiddenLayers = 1;
        //    NumNeuronsPerLayer = new int[] { 1 };
        //    ActivationFunctionPerLayer = new Activation[] { keras.activations.Linear };
        //}

        public void Train(float[] trainX, float[] trainY, double[,] testX = null, double[,] testY = null)
        {
            tf.enable_eager_execution();

            this.trainX = np.array(trainX);
            this.trainY = np.array(trainY);
            
            if (testX != null && testY != null)
            {
                this.testX = np.array(testX);
                this.testY = np.array(testY);
            }
            
            var layers = new LayersApi();

            inputs = keras.Input(shape: 1); // 0 or 1 ??

            outputs = layers.Dense(NumNeuronsPerLayer[0], ActivationFunctionPerLayer[0]).Apply(inputs);

            for (int i = 1; i < NumHiddenLayers; i++)
            {
                outputs = layers.Dense(NumNeuronsPerLayer[i], ActivationFunctionPerLayer[i]).Apply(outputs);
            }

            outputs = layers.Dense(1).Apply(outputs);

            // build keras model
            model = keras.Model(inputs, outputs, name: "current_model");
            // show model summary
            model.summary();

            // compile keras model into tensorflow's static graph
            model.compile(loss: LossFunction,
                optimizer: Optimizer,
                metrics: new[] { "accuracy" });
            
            // train the model
            model.fit(this.trainX, this.trainY, batch_size: BatchSize, epochs: Epochs);

            // evaluate the model
            if (testX != null && testY != null)
            {
                model.evaluate(this.testX, this.testY, batch_size: BatchSize);
            }
            
            // save and serialize model
            model.save("current_model");

            // recreate the exact same model purely from the file:
            // model = keras.models.load_model("path_to_my_model");
        }

        public void Predict(float[,] data)
        {
            var npData = np.array(data);
            // predict output of new data
            //var pred1 = model.predict(this.trainX);
            var pred2 = model.Apply(npData, training: false);
        }

        float[,] Transpose(float[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            float[,] result = new float[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }
    }
}
