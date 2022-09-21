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
    public class FeedForwardNeuralNetwork : Model
    {
        public INormalization Normalization { get; set; }

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

        public FeedForwardNeuralNetwork(ModelArgs? args = null) : base(args: new ModelArgs())
        {
            Optimizer = keras.optimizers.Adam();
            LearningRate = 0.001;
            BatchSize = -1;
            Epochs = 1;
            DisplayStep = 100;
            NumHiddenLayers = 1;
            NumNeuronsPerLayer = new int[] { 1 };
            ActivationFunctionPerLayer = new Activation[] { keras.activations.Linear };
            Layer = new Layer[1] { keras.layers.Dense(NumNeuronsPerLayer[0], ActivationFunctionPerLayer[0]) };
        }

        public void Train(double[,] trainX, double[,] trainY, double[,] testX = null, double[,] testY = null)
        {
            tf.enable_eager_execution();

            if (Normalization != null) 
            {   
                if (testX != null)
                {
                    var trainAndTestX = new double[trainX.GetLength(0) + testX.GetLength(0),trainX.GetLength(1)];
                    trainX.CopyTo(trainAndTestX, 0);
                    testX.CopyTo(trainAndTestX, trainX.Length);
                    Normalization.Initialize(trainAndTestX, dim : 1);
                    //
                    trainX = Normalization.Normalize(trainX);
                    testX = Normalization.Normalize(testX);
                }
                else
                {
                    Normalization.Initialize(trainX, dim: 1);
                    trainX = Normalization.Normalize(trainX);
                }
            }

            this.trainX = np.array(ToFloat(trainX));
            this.trainY = np.array(ToFloat(trainY));
            
            if (testX != null && testY != null)
            {
                this.testX = np.array(ToFloat(testX));
                this.testY = np.array(ToFloat(testY));
            }
            
            var layers = new LayersApi();

            inputs = keras.Input(shape: trainX.GetLength(1)); // 0 or 1 ??

            outputs = layers.Dense(NumNeuronsPerLayer[0], ActivationFunctionPerLayer[0]).Apply(inputs);

            for (int i = 1; i < NumHiddenLayers; i++)
            {
                outputs = layers.Dense(NumNeuronsPerLayer[i], ActivationFunctionPerLayer[i]).Apply(outputs);
            }
            
            outputs = layers.Dense(trainY.GetLength(1)).Apply(outputs);

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

        public void Predict(double[,] data)
        {
            if (Normalization != null)
            {
                (data) = Normalization.Normalize(data);
            }
            var npData = np.array(ToFloat(data));
            // predict output of new data
            //var pred1 = model.predict(this.trainX);
            var pred2 = model.Apply(npData, training: false);
        }

        public void Gradient(double[,] data)
        {
            if (Normalization != null)
            {
                (data) = Normalization.Normalize(data);
            }
            var npData  = np.array(ToFloat(data));
            //trainX = list(trainX);
            npData = (NDArray)tf.constant(npData);
            using var tape = tf.GradientTape();
            {
                tape.watch(npData);
                // Forward pass.
                var pred = model.Apply(npData, training: false);
                //var loss = cross_entropy_loss(pred, batch_y);

                //var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
                //var loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels: tf.cast(trainY, tf.int64), logits: pred));
                var loss = tf.pow(pred - trainY, 2.0f);
                //return tf.reduce_mean(loss);

                // Compute gradients.
                //var gradients1 = g.gradient(pred, model.trainable_variables);
                var gradients2 = tape.gradient(pred, npData);
            }
        }

        float[,] ToFloat(double[,] matrix)
        {
            float[,] floatMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
            for (int i = 0; matrix.GetLength(0) > i; i++)
            {
                for (int j = 0; matrix.GetLength(1) > j; j++)
                {
                    floatMatrix[i,j] = (float)matrix[i,j];
                }
            }
            return floatMatrix;
        }
    }
}
