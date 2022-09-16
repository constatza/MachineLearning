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
    public class FeedForwardNeuralNetwork : Model
    {
        string Preprocessing { get; set; }

        //Optimization Parameters
        public string Optimizer { get; set; }
        public double LearningRate { get; set; }
        public int BatchSize { get; set; }
        public int Epochs { get; set; }
        public int DisplayStep { get; set; }
        public Layer[] Layer { get; private set; }

        //Neural Network Architecture
        public int NumHiddenLayers { get; set; }
        public int[] NumNeuronsPerLayer { get; set; }
        public Activation[] ActivationFunctionPerLayer { get; set; }

        //ModelArgs Args { get; set; }

        public FeedForwardNeuralNetwork(ModelArgs? args = null) : base(args: new ModelArgs())
        {
            //InitializeParameters();
            Preprocessing = "minmax";
            Optimizer = "SGD";
            LearningRate = 0.01;
            BatchSize = 50;
            Epochs = 100;
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

        public void TrainNetwork(double[] trainX, double[] trainY)
        {
            tf.enable_eager_execution();

            var npTrainX = np.array(trainX);
            var npTrainY = np.array(trainY);
            var train_data = tf.data.Dataset.from_tensor_slices(npTrainX, npTrainY);
            train_data = train_data.repeat()
                //.shuffle(5000)
                .batch(BatchSize)
                //.prefetch(1)
                .take(Epochs);

            var neuralNetworkArgs = new ModelArgs();

            var neuralNetwork = new Model(neuralNetworkArgs);

            var optimizer = keras.optimizers.SGD((float)LearningRate);

            Layer = new Layer[NumHiddenLayers + 1];

            var layers = keras.layers;

            for (int i = 0; i < NumHiddenLayers; i++)
            {
                Layer[i] = layers.Dense(NumNeuronsPerLayer[i], ActivationFunctionPerLayer[i]);
            }

            if (trainY.Rank == 1)
            {
                Layer[NumHiddenLayers] = layers.Dense(1); // 0 or 1 ??
            }
            else
            {
                Layer[NumHiddenLayers] = layers.Dense(trainY.GetLength(1));
            }
            StackLayers(Layer);

            Func<Tensor, Tensor, Tensor> cross_entropy_loss = (x, y) =>
            {
                y = tf.cast(y, tf.int64);
                //var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
                var loss = tf.reduce_sum(tf.pow(x - y, 2.0f)) / (2.0f * trainY.GetLength(0));
                //return tf.reduce_mean(loss);
                return loss;
            };

            Func<Tensor, Tensor, Tensor> accuracy = (y_pred, y_true) =>
            {
                // Predicted class is the index of highest score in prediction vector (i.e. argmax).
                var correct_prediction = tf.equal(tf.math.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
            };

            Action<Tensor, Tensor> run_optimization = (x, y) =>
            {
                // Wrap computation inside a GradientTape for automatic differentiation.
                using var g = tf.GradientTape();
                // Forward pass.
                var pred = Apply(x, training: true);
                var loss = cross_entropy_loss(pred, y);

                // Compute gradients.
                var gradients = g.gradient(loss, trainable_variables);

                // Update W and b following gradients.
                optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));
            };

            // Run training for the given number of steps.
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.
                run_optimization(batch_x, batch_y);

                if (step % DisplayStep == 0)
                {
                    var pred = Apply(batch_x, training: true);
                    var loss = cross_entropy_loss(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }
        }

        //public class FeedForwardNeuralNetArgs : ModelArgs
        //{
        //    /// <summary>
        //    /// 1st layer number of neurons.
        //    /// </summary>

        //    public int NeuronOfHidden1 { get; set; }
        //    public Activation Activation1 { get; set; }

        //    /// <summary>
        //    /// 2nd layer number of neurons.
        //    /// </summary>
        //    public int NeuronOfHidden2 { get; set; }
        //    public Activation Activation2 { get; set; }

        //    public int NumClasses { get; set; }
        //}

        public double[] Predict(double[] X)
        {
            var npX = np.array(X);
            var npPrediction = Apply(npX, training: false);
            double[] prediction = new double[1];
            return prediction;
        }
    }
}
